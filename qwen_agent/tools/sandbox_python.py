# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A sandbox-style Python code execution tool for Qwen-Agent.

This tool is a *pure Qwen-Agent* replacement for the Docker-based
``code_interpreter``: it runs Python via ``exec`` in a restricted namespace
(see ``safe_builtins`` + ``_safe_import``), with no Docker dependency.

Design (follows Qwen-Agent's tool conventions exactly):
  * Subclasses ``BaseToolWithFileAccess`` and sets ``file_access = True`` so the
    framework auto-passes any files/images that appeared in the conversation
    (``extract_files_from_messages(..., include_images=True)``) into ``call()``
    via the ``files`` argument. They are materialized into a per-task
    ``work_dir`` — the Qwen-Agent "files live in the work dir" model: the model
    reads them by filename (``PIL.Image.open('x.png')``, ``pd.read_csv(...)``).
  * Generated images (``plt.savefig`` / new files in work_dir / open matplotlib
    figures) are returned as ``List[ContentItem(image=...)]`` so the framework
    feeds them back to the model as a multimodal tool result.
  * Execution is sandboxed: a restricted ``__builtins__``, an import whitelist,
    stdout/stderr capture, a timeout, and a per-task working directory.

Per-task state (the kernel is reused across calls within one task, so variables
persist between turns, just like a Jupyter kernel) is keyed by ``task_id``.
"""

import base64
import contextlib
import glob as _glob
import io
import json
import os
import re
import shutil
import signal
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import json5

from qwen_agent.llm.schema import ContentItem
from qwen_agent.log import logger
from qwen_agent.tools.base import BaseToolWithFileAccess, register_tool
from qwen_agent.utils.utils import save_url_to_local_work_dir


# ---------------------------------------------------------------------------
# Security: import whitelist + safe builtins (ported from AgentFlow ds_tool).
# ---------------------------------------------------------------------------

ALLOWED_MODULES = {
    'pandas', 'numpy', 'scipy', 'statsmodels', 'patsy',
    'sklearn', 'joblib', 'xgboost', 'lightgbm', 'imblearn',
    'cv2', 'PIL',
    'sys', 'os', 'io', 'pathlib', 'json', 'csv', 'pickle', 'glob', 'shutil',
    'datetime', 'time', 'dateutil', 'calendar', 'pytz',
    'collections', 'itertools', 'functools', 'operator', 'heapq', 'bisect',
    'copy', 'enum',
    're', 'string', 'textwrap', 'difflib', 'unicodedata',
    'math', 'random', 'statistics', 'decimal', 'fractions',
    'warnings', 'logging', 'pprint', 'uuid', 'hashlib', 'typing', 'dataclasses',
    'matplotlib', 'seaborn', 'plotly', 'altair',
    'urllib', 'html', 'xml',
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root_module = name.split('.')[0]
    if root_module not in ALLOWED_MODULES:
        raise ImportError(f"Security: Import of module '{root_module}' is blocked.")
    return __import__(name, globals, locals, fromlist, level)


_SAFE_BUILTINS = {
    '__import__': _safe_import,
    'abs': abs, 'min': min, 'max': max, 'sum': sum, 'round': round, 'pow': pow,
    'divmod': divmod, 'all': all, 'any': any,
    'len': len, 'range': range, 'enumerate': enumerate, 'zip': zip,
    'map': map, 'filter': filter, 'iter': iter, 'next': next,
    'slice': slice, 'reversed': reversed, 'sorted': sorted,
    'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'frozenset': frozenset,
    'str': str, 'int': int, 'float': float, 'bool': bool, 'complex': complex,
    'bytes': bytes, 'bytearray': bytearray,
    'type': type, 'isinstance': isinstance, 'issubclass': issubclass,
    'callable': callable, 'hash': hash, 'id': id, 'object': object,
    'getattr': getattr, 'setattr': setattr, 'hasattr': hasattr, 'delattr': delattr,
    'vars': vars, 'dir': dir,
    'print': print, 'repr': repr, 'ascii': ascii, 'format': format,
    'chr': chr, 'ord': ord, 'bin': bin, 'hex': hex, 'oct': oct,
    'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
    'KeyError': KeyError, 'IndexError': IndexError, 'NameError': NameError,
    'AttributeError': AttributeError, 'ImportError': ImportError,
    'RuntimeError': RuntimeError, 'ZeroDivisionError': ZeroDivisionError,
    'StopIteration': StopIteration, 'FileNotFoundError': FileNotFoundError,
    'OSError': OSError, 'AssertionError': AssertionError,
    'FileExistsError': FileExistsError, 'NotImplementedError': NotImplementedError,
}


# ---------------------------------------------------------------------------
# Per-task execution kernel (state persists across calls within one task).
# ---------------------------------------------------------------------------

class _Kernel:
    """A persistent exec namespace for one task (like a Jupyter kernel)."""

    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        self.globals: Dict[str, Any] = {}
        self._init_namespace()

    def _init_namespace(self):
        import datetime
        import math
        import random
        import collections
        import itertools
        import warnings
        import re as _re

        self.globals.update({
            '__builtins__': _SAFE_BUILTINS,
            '__name__': '__main__',
            'os': os, 'sys': __import__('sys'), 'io': io, 'json': json,
            're': _re, 'math': math, 'random': random,
            'datetime': datetime, 'collections': collections,
            'itertools': itertools, 'warnings': warnings,
            'Path': Path, 'glob': _glob, 'shutil': shutil,
        })
        # Optional data-science libs (graceful if absent).
        try:
            import numpy as np
            self.globals['np'] = np
        except ImportError:
            pass
        try:
            import pandas as pd
            self.globals['pd'] = pd
        except ImportError:
            pass
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            self.globals['plt'] = plt
            try:
                import seaborn as sns
                self.globals['sns'] = sns
            except ImportError:
                pass
        except ImportError:
            plt = None  # noqa: F841
        try:
            import cv2
            self.globals['cv2'] = cv2
        except ImportError:
            pass
        try:
            import scipy
            self.globals['scipy'] = scipy
        except ImportError:
            pass
        try:
            import sklearn
            self.globals['sklearn'] = sklearn
        except ImportError:
            pass

        warnings.filterwarnings('ignore')


_KERNELS: Dict[str, _Kernel] = {}


def _materialize_files(files: List[str], work_dir: str) -> List[str]:
    """Copy/download remote files & resolve local image paths into work_dir.

    Returns the list of local file paths now present in work_dir.
    For local paths that already exist, we symlink/copy so the model can open
    them by basename inside the kernel's cwd (== work_dir).
    """
    os.makedirs(work_dir, exist_ok=True)
    local_paths: List[str] = []
    for f in files or []:
        if not f:
            continue
        if f.startswith(('http://', 'https://')):
            try:
                p = save_url_to_local_work_dir(f, work_dir)
                local_paths.append(p)
            except Exception:
                logger.warning(f'Failed to download {f}')
        else:
            src = os.path.abspath(os.path.expanduser(f))
            if os.path.exists(src):
                dst = os.path.join(work_dir, os.path.basename(src))
                if os.path.abspath(src) != os.path.abspath(dst):
                    try:
                        if os.path.exists(dst) or os.path.islink(dst):
                            os.remove(dst)
                        os.symlink(src, dst)
                    except OSError:
                        shutil.copy(src, dst)
                local_paths.append(dst)
            else:
                logger.warning(f'File not found, skipped: {f}')
    return local_paths


class _TimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _TimeoutError('Code execution exceeded the time limit.')


# File extensions considered "generated images".
_IMAGE_GLOB_EXTS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.webp')


@register_tool('sandbox_python')
class SandboxPython(BaseToolWithFileAccess):
    """Sandbox-style Python execution (no Docker) for Qwen-Agent.

    Parameters:
      * ``code`` (str, required): Python code to execute.
      * ``return_vars`` (list[str], optional): variable names to inspect after.
    """

    description = (
        'Execute Python code in a sandboxed namespace for data analysis / '
        'image processing / calculation. Pre-installed: pandas, numpy, scipy, '
        'sklearn, matplotlib, seaborn, cv2, PIL. '
        'Any files or images attached to the conversation are available in the '
        'current working directory — open them by filename, e.g. '
        "`from PIL import Image; img = Image.open('dat_pat_q01.png')` or "
        "`import cv2; arr = cv2.imread('dat_pat_q01.png')`. "
        'You can save figures with `plt.savefig(\'out.png\')`; saved images are '
        'returned to you. Do NOT load files from hard-coded absolute paths.'
    )
    parameters = {
        'type': 'object',
        'properties': {
            'code': {
                'type': 'string',
                'description': 'The Python code to execute.',
            },
            'return_vars': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': 'Optional: variable names to inspect after execution.',
            },
        },
        'required': ['code'],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.timeout: int = int(self.cfg.get('timeout', 120))
        self.root_work_dir: str = self.cfg.get('root_work_dir', self.work_dir)
        os.makedirs(self.root_work_dir, exist_ok=True)

    # ------------------------------------------------------------------ API

    def call(self,
             params: Union[str, dict],
             files: List[str] = None,
             **kwargs) -> Union[str, List[ContentItem]]:
        # 1) Materialize conversation files/images into work_dir (Qwen-Agent).
        #    The framework passes any files/images seen in the conversation via
        #    `files`; they land in the per-task work dir so the model can open
        #    them by filename.
        task_id = str(kwargs.get('task_id') or self.cfg.get('task_id') or uuid.uuid4().hex)
        work_dir = os.path.join(self.root_work_dir, f'task_{task_id}')
        os.makedirs(work_dir, exist_ok=True)
        _materialize_files(files, work_dir)

        args = self._verify_json_format_args(params)
        code = args.get('code', '') or ''
        return_vars = args.get('return_vars')

        if not code.strip():
            return 'No code provided.'

        # 2) Get/reuse the per-task kernel (state persists across turns).
        kernel = _KERNELS.get(task_id)
        if kernel is None:
            kernel = _Kernel(work_dir)
            _KERNELS[task_id] = kernel

        # 3) Execute with stdout/stderr capture + timeout.
        text_result, generated_images = self._exec(kernel, code, return_vars)

        # 4) Build the return: a text ContentItem + one ContentItem per image.
        items: List[ContentItem] = [ContentItem(text=text_result)]
        for img_path in generated_images:
            items.append(ContentItem(image=os.path.abspath(img_path)))
        return items

    # -------------------------------------------------------------- internals

    def _exec(self, kernel: _Kernel, code: str,
              return_vars: Optional[List[str]]) -> (str, List[str]):
        import warnings
        plt = kernel.globals.get('plt')

        stdout_capture = io.StringIO()
        error_message = None
        before_files = set(_list_image_files(kernel.work_dir))

        # signal-based timeout (main thread only).
        use_alarm = (os.name == 'posix') and (threading_current_is_main())
        old_handler = None
        if use_alarm:
            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(self.timeout)

        try:
            with contextlib.redirect_stdout(stdout_capture), \
                    contextlib.redirect_stderr(stdout_capture):
                with _temporary_chdir(kernel.work_dir):
                    exec(code, kernel.globals)
        except _TimeoutError:
            error_message = f'Timeout: code execution exceeded {self.timeout}s.'
        except Exception:
            error_message = traceback.format_exc()
        finally:
            if use_alarm:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
            if plt is not None:
                try:
                    plt.close('all')
                except Exception:
                    pass

        # Collect generated images: (a) newly written image files, (b) open figs.
        generated: List[str] = []
        for path in _list_image_files(kernel.work_dir):
            if path in before_files:
                continue
            if os.path.islink(path):
                continue
            generated.append(path)

        if plt is not None and not generated:
            for fnum in plt.get_fignums():
                fig = plt.figure(fnum)
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                outp = os.path.join(kernel.work_dir, f'fig_{uuid.uuid4().hex}.png')
                with open(outp, 'wb') as fout:
                    fout.write(buf.read())
                generated.append(outp)

        # Compose text result.
        stdout = stdout_capture.getvalue().strip()
        MAX_CHARS = 5000
        if len(stdout) > MAX_CHARS:
            stdout = stdout[:MAX_CHARS] + f'\n\n[System Warning]: Output truncated (total {len(stdout)} chars).'

        parts: List[str] = []
        if error_message:
            parts.append(f'❌ Execution Error:\n{error_message}')
        if stdout:
            parts.append(f'📄 Standard Output:\n{stdout}')
        if return_vars and not error_message:
            caps = []
            for name in return_vars:
                if name in kernel.globals:
                    val = kernel.globals[name]
                    try:
                        import pandas as pd
                        if isinstance(val, pd.DataFrame):
                            caps.append(f'{name} (shape={val.shape}):\n{val.head(5).to_markdown(index=False)}')
                            continue
                    except Exception:
                        pass
                    caps.append(f'{name} = {val!r}')
            if caps:
                parts.append('📦 Variables Inspection:\n' + '\n'.join(caps))
        if not parts and not generated:
            parts.append('✅ Execution successful (no output printed).')

        return '\n\n'.join(parts), generated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_image_files(work_dir: str) -> List[str]:
    out: List[str] = []
    try:
        for ext in _IMAGE_GLOB_EXTS:
            out.extend(_glob.glob(os.path.join(work_dir, ext)))
    except Exception:
        pass
    return sorted(out)


def threading_current_is_main() -> bool:
    import threading
    return threading.current_thread() is threading.main_thread()


@contextlib.contextmanager
def _temporary_chdir(path: str):
    cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)
