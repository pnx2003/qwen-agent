"""Run a multimodal MCQ benchmark with a pure-Qwen-Agent code agent.

This is the Qwen-Agent analogue of AgentFlow's
``configs/infer/mybenchmark/mybenchmark_tool_gpt5.5.json``:
  * one tool only — ``sandbox_python`` (a Docker-free, sandbox-style Python
    executor that also injects task images as ``image_clue`` + files in the
    work dir, compatible with AgentFlow's ``ds_run_python`` input contract);
  * AgentFlow-format JSONL input (``query``/``question``/``images``/``image``/
    ``image_path``/``answer``/``task_id``/``question_id`` ...);
  * ``my_mcq``-style option extraction + scoring;
  * saves results.jsonl + trajectories.

Usage (from the Qwen-Agent repo root, in an env that has qwen-agent + ds libs):

    python examples/benchmark_sandbox_python.py \\
        --config /cpfs01/pnx/AgentFlow/configs/infer/mybenchmark/mybenchmark_tool_gpt5.5.json

Or pass overrides directly. See ``argparse`` below.

NOTE on vision: if the model endpoint supports image input (a VLM), set
``--vision`` so the task image is also shown to the model in the user message.
If it is a text-only endpoint, leave it off — the model can still process the
image by writing code (``image_clue[0]`` or ``PIL.Image.open(...)``).
"""
import argparse
import base64
import glob
import json
import mimetypes
import os
import re
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:  # pragma: no cover
    ZoneInfo = None
from typing import Any, Dict, List, Optional

# Make the local Qwen-Agent importable when running from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_agent.agents import FnCallAgent  # noqa: E402
from qwen_agent.llm.schema import ContentItem, Message  # noqa: E402
from qwen_agent.tools.sandbox_python import SandboxPython, _KERNELS  # noqa: E402


# ---------------------------------------------------------------------------
# MCQ answer extraction + scoring (mirrors AgentFlow my_mcq).
# ---------------------------------------------------------------------------

def extract_mcq_option(text: str) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    lines = text.split('\n')

    for line in reversed(lines):
        m = re.match(r'^\*{0,3}\s*[Aa]nswer\*{0,3}\s*[:：]\s*([A-Ja-j])\s*$', line.strip())
        if m:
            return m.group(1).upper()
    for line in reversed(lines):
        m = re.match(r'^\*{0,3}\s*[Aa]nswer\*{0,3}\s*[:：]\s*([A-Ja-j])', line.strip())
        if m:
            return m.group(1).upper()

    marker_patterns = [
        r'(?i)(?:the\s+)?answer\s+is\s+([A-Ja-j])(?![a-zA-Z])',
        r'(?i)(?:choice|option|select|choose)\s+([A-Ja-j])(?![a-zA-Z])',
        r'(?i)(?:correct\s+)?answer\s*[:：]\s*([A-Ja-j])(?![a-zA-Z])',
    ]
    for p in marker_patterns:
        ms = re.findall(p, text)
        if ms:
            return ms[-1].upper()

    for p in [r'\*\*([A-Ja-j])\*\*', r'__([A-Ja-j])__', r'\*([A-Ja-j])\*']:
        ms = re.findall(p, text)
        if ms:
            return ms[-1].upper()

    m = re.match(r'^\s*([A-Ja-j])(?![a-zA-Z])\s*$', text)
    if m:
        return m.group(1).upper()

    tail = ' '.join(lines[-5:])
    ms = re.findall(r'(?<![a-zA-Z])([A-Ja-j])(?![a-zA-Z])', tail)
    if ms:
        return ms[-1].upper()
    return None


def extract_final_answer(text: str) -> str:
    if not text:
        return ''
    text = text.replace('\r\n', '\n')
    lines = text.split('\n')
    for line in lines:
        s = line.strip()
        if not s:
            continue
        m = re.match(r'(?i)^\*{0,3}\s*answer\*{0,3}\s*[:：\-]\s*(.*)$', s)
        if m:
            c = m.group(1).strip().strip('*`').strip()
            if c:
                return c
    return text.strip()


def score_mcq(predicted: str, ground_truth: str) -> Dict[str, Any]:
    pred = extract_mcq_option(predicted)
    gt = extract_mcq_option(ground_truth)
    if gt is None:
        g = ground_truth.strip().upper()
        gt = g if len(g) == 1 and g in 'ABCDEFGHIJ' else None
    if pred and gt:
        correct = pred.upper() == gt.upper()
        return {'score': 1.0 if correct else 0.0, 'pred': pred, 'gt': gt, 'correct': correct}
    return {'score': 0.0, 'pred': pred, 'gt': gt, 'correct': False,
            'error': 'could not extract option'}


# ---------------------------------------------------------------------------
# AgentFlow-compatible JSONL loading.
# ---------------------------------------------------------------------------

def _resolve_image_to_b64_or_path(img: str, source_dir: Optional[str]) -> str:
    """Return a path (preferred) or data URI for an image field value."""
    txt = (img or '').strip()
    if not txt:
        return ''
    if txt.startswith(('http://', 'https://', 'data:')):
        return txt
    candidate = Path(txt)
    if not candidate.is_absolute() and source_dir:
        candidate = Path(source_dir) / txt
    if candidate.is_file():
        return str(candidate)
    # Fall back: assume raw base64.
    return txt


def load_tasks(path: str) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    source_dir = str(Path(path).resolve().parent)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # task_id is expected to be unique per question in the source data
            # (e.g. dat_pat_spatial_folding_01 .. _10). It is used as the sandbox
            # work_dir and kernel key (task_<id>), so it MUST be unique per
            # question — otherwise questions sharing a task_id collapse into one
            # work dir and overwrite each other's generated files.
            tid = str(item.get('id') or item.get('task_id')
                      or item.get('question_id') or item.get('key_question') or '')
            question = (item.get('query') or item.get('question')
                        or item.get('input') or item.get('prompt') or '')
            answer = (item.get('answer') or item.get('ground_truth')
                      or item.get('expected') or item.get('gt'))

            raw_images = (item.get('images') or item.get('image')
                          or item.get('image_byte') or item.get('image_path')
                          or item.get('images_list') or [])
            if isinstance(raw_images, str):
                raw_images = [raw_images]
            images = [_resolve_image_to_b64_or_path(im, source_dir)
                      for im in raw_images if im]

            tasks.append({
                'id': tid,
                'task_id': str(item.get('task_id') or ''),
                'question_id': str(item.get('question_id') or ''),
                'question': question,
                'answer': answer,
                'images': images,
                'choices': item.get('choices') or item.get('options') or [],
                'source_dir': source_dir,
                'metadata': {k: v for k, v in item.items()
                             if k not in {'query', 'question', 'input', 'prompt',
                                          'answer', 'ground_truth', 'expected', 'gt',
                                          'images', 'image', 'image_byte', 'image_path',
                                          'images_list', 'task_id', 'question_id',
                                          'id', 'key_question', 'source_dir',
                                          'choices', 'options'}},
            })
    return tasks


def _matches_task_id(task: Dict[str, Any], wanted: set) -> bool:
    """A task matches if its task_id equals a wanted value, or starts with
    ``<wanted>_`` (prefix match).

    task_id is per-question unique in the source data (e.g.
    ``dat_pat_spatial_folding_01``), so:
      * ``--task_ids dat_pat_spatial_folding_02`` -> exact match -> single question;
      * ``--task_ids dat_pat_spatial_folding``    -> prefix match -> all 10 questions.
    """
    task_id = task.get('task_id') or ''
    for w in wanted:
        if task_id == w or task_id.startswith(w + '_'):
            return True
    return False


def build_user_content(task: Dict[str, Any], vision: bool) -> List[ContentItem]:
    """Build the user message: question text + image.

    The image is ALWAYS embedded as a ContentItem(image=...) so the framework's
    file-access channel routes it to the tool's `files` (via
    extract_files_from_messages(include_images=True)). When ``vision`` is True
    the VLM also sees the image directly. When False, a text note tells the
    model the image filename so it can open it by name in the code tool.
    """
    question_text = task['question']
    # Ensure multiple-choice options are visible to the model. If the task
    # carries a `choices` dict (e.g. angle-ranking orderings) but the question
    # text doesn't already list them, append them.
    choices = task.get('choices')
    if choices and 'Options:' not in question_text and 'options' not in question_text.lower():
        if isinstance(choices, dict):
            opts = '\n'.join(f'  {k}. {v}' for k, v in choices.items())
        else:
            opts = '\n'.join(f'  {i+1}. {c}' for i, c in enumerate(choices))
        question_text = question_text.rstrip() + '\n\nOptions:\n' + opts

    content: List[ContentItem] = [ContentItem(text=question_text)]
    filenames = []
    for img in task['images']:
        if not img:
            continue
        content.append(ContentItem(image=img))
        filenames.append(os.path.basename(img))
    if not vision and filenames:
        content.append(ContentItem(
            text='\n\n(题目图片已放入代码工具的工作目录，文件名为: '
                 + ', '.join(filenames) + '。你无法直接看到图片，必须调用 '
                 'sandbox_python 工具，用 PIL/cv2 读取并分析图片后再作答。'
                 "例如: from PIL import Image; img = Image.open('"
                 + filenames[0] + "')"))
    return content


def _looks_like_local_path(s: str) -> bool:
    return s.startswith('/') or (not s.startswith(('http://', 'https://', 'data:'))
                                 and Path(s).is_file())


# ---------------------------------------------------------------------------
# Agent construction.
# ---------------------------------------------------------------------------

MCQ_SUFFIX = (
    '\n\n## Answer Format (IMPORTANT)\n'
    'This is a multiple-choice question. After your reasoning, you MUST output '
    'your final answer as a single uppercase option letter on its own line:\n\n'
    'Answer: X\n\n'
    'where X is the letter (e.g. A, B, C, D). Do NOT wrap it in markdown, do '
    'NOT embed it in a sentence, and output nothing after that line.'
)


def build_agent(cfg: Dict[str, Any]) -> FnCallAgent:
    llm_cfg: Dict[str, Any] = {
        'model': cfg['model_name'],
        'model_server': cfg['base_url'],
        'api_key': cfg['api_key'],
    }
    if cfg.get('vision'):
        llm_cfg['model_type'] = 'qwenvl_oai'
    llm_cfg['generate_cfg'] = {
        'max_retries': int(cfg.get('max_retries', 3)),
        # Use the endpoint's native OpenAI tool-calling API (tools=... +
        # tool_calls) instead of Qwen-Agent's prompt-template fncall. Required
        # for models like gpt-5.4 that support native function calling, so the
        # model's tool calls are recognized instead of emitted as raw text.
        'use_raw_api': bool(cfg.get('use_raw_api', True)),
    }
    if cfg.get('max_input_tokens'):
        llm_cfg['generate_cfg']['max_input_tokens'] = int(cfg['max_input_tokens'])

    vision = bool(cfg.get('vision'))
    # The system prompt only specifies (1) how to call the code tool and (2) the
    # output format. It deliberately contains NO task-type guidance (e.g. angle
    # ranking) — that was misleading on other task types like spatial folding.
    if vision:
        system = (cfg.get('system_prompt') or
                  'You are an expert problem solver. Look at the provided image, '
                  'reason carefully, and give your final answer. A Python code '
                  'tool is available if you want to inspect or verify, but you '
                  'decide whether it helps.')
    else:
        system = (cfg.get('system_prompt') or
                  'You are an expert problem solver. Solve the question, using the '
                  'Python code tool to analyze images, compute, or verify when '
                  'helpful. Then give your final answer.')
    # If the config's prompt mentions the AgentFlow tool name, clarify the
    # actual tool name so the model calls the right function.
    if 'ds_run_python' in system and 'sandbox_python' not in system:
        system += ('\n\n(Note: the available code tool is named `sandbox_python`, '
                   'use it the same way you would `ds_run_python`.)')

    # Text-only path: the model cannot see the image, so the only way to access
    # it is through the code tool. This block is a tool-call instruction (how to
    # reach the image file), not task-type guidance. In vision mode the model
    # sees the image directly, so it is omitted.
    if not vision:
        system += (
            '\n\n## Image Access\n'
            'You CANNOT see the task image directly. The image file is in your '
            "code tool's working directory. Call `sandbox_python` and write "
            'Python to load and inspect it, e.g.:\n'
            "```python\nfrom PIL import Image\nimport numpy as np\n"
            "img = Image.open('<filename>')   # filename shown in the task\n"
            "print(img.size, np.array(img).shape)\n```")

    system += MCQ_SUFFIX

    tool_cfg = {
        'name': 'sandbox_python',
        'timeout': int(cfg.get('sandbox_timeout', 300)),
        'root_work_dir': os.path.join(cfg['output_dir'], 'sandbox_workdir'),
    }
    bot = FnCallAgent(
        llm=llm_cfg,
        function_list=[tool_cfg],
        system_message=system,
        name='benchmark_code_agent',
        description='A code-using agent for multimodal MCQ benchmarks.',
    )
    return bot


# ---------------------------------------------------------------------------
# Run loop.
# ---------------------------------------------------------------------------

def run_task(bot: FnCallAgent, task: Dict[str, Any], max_turns: int,
             vision: bool) -> Dict[str, Any]:
    # Reset per-task sandbox kernel state so each task starts clean.
    _KERNELS.pop(str(task['id']), None)

    user_content = build_user_content(task, vision=vision)
    messages: List[Message] = [Message(role='user', content=user_content)]

    # IMPORTANT: use the public bot.run() (not _run()). run() injects the
    # agent's system_message (tool descriptions + answer-format) into the
    # conversation — without it the model never learns the tools exist and
    # never calls them. run() also yields the growing response list each turn,
    # which we accumulate to reconstruct the full trajectory.
    final_text = ''
    all_messages: List[Message] = [Message(role='user', content=user_content)]
    try:
        # task_id flows through run() -> _run() -> _call_tool() -> tool.call()
        # so the sandbox keys its kernel per task.
        for response in bot.run(messages, task_id=task['id']):
            if response:
                all_messages = response  # growing list of all messages so far
                last = response[-1]
                if last.role == 'assistant' and isinstance(last.content, str) and last.content:
                    final_text = last.content
    except Exception as e:
        traceback.print_exc()
        final_text = f'AGENT_ERROR: {e}'

    answer = extract_final_answer(final_text)
    return {
        'final_text': final_text,
        'predicted_answer': answer,
        'trajectory': _serialize_trajectory(all_messages),
    }


def _serialize_trajectory(messages: List[Message]) -> List[Dict[str, Any]]:
    out = []
    for m in messages:
        content = m.content
        if isinstance(content, list):
            content = [{'type': getattr(c, 'type', None),
                        'value': getattr(c, 'value', None)} for c in content]
        # function_call is a FunctionCall pydantic object — make it JSON-safe.
        fc = m.function_call
        if fc is not None:
            fc = {'name': getattr(fc, 'name', None),
                  'arguments': getattr(fc, 'arguments', None)}
        out.append({'role': m.role, 'content': content,
                    'name': m.name, 'function_call': fc})
    return out


# ---------------------------------------------------------------------------
# Config + main.
# ---------------------------------------------------------------------------

def load_config(path: Optional[str], overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    # Map AgentFlow field names -> our fields.
    cfg.setdefault('model_name', cfg.get('model_name', ''))
    cfg.setdefault('api_key', cfg.get('api_key', ''))
    cfg.setdefault('base_url', cfg.get('base_url', ''))
    cfg.setdefault('data_path', cfg.get('data_path', ''))
    cfg.setdefault('output_dir', cfg.get('output_dir', 'infer_results/qwen_agent'))
    cfg.setdefault('max_turns', int(cfg.get('max_turns', 40)))
    cfg.setdefault('sandbox_timeout', int(cfg.get('sandbox_timeout', 300)))
    # Normalize system_prompt: AgentFlow configs store it as a list of lines.
    sp = cfg.get('system_prompt')
    if isinstance(sp, list):
        cfg['system_prompt'] = '\n'.join('' if x is None else str(x) for x in sp)
    elif sp is None:
        cfg['system_prompt'] = ''
    cfg.update(overrides)
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=None, help='AgentFlow-style JSON config')
    p.add_argument('--data_path', default=None)
    p.add_argument('--output_dir', default=None)
    p.add_argument('--model_name', default=None)
    p.add_argument('--api_key', default=None)
    p.add_argument('--base_url', default=None)
    p.add_argument('--max_turns', type=int, default=None)
    p.add_argument('--max_tasks', type=int, default=None)
    p.add_argument('--task_ids', default=None, help='comma-separated task ids')
    p.add_argument('--vision', action='store_true',
                   help='model endpoint supports image input (VLM)')
    p.add_argument('--no-vision', dest='vision', action='store_false')
    p.set_defaults(vision=None)
    p.add_argument('--no_timestamp', action='store_true',
                   help='do not append a timestamp subdirectory to output_dir '
                        '(by default each run is archived under '
                        'output_dir/<YYYYMMDD_HHMMSS> so reruns never overwrite)')
    p.add_argument('--resume', default=None,
                   help='resume from an existing result directory: skip tasks whose '
                        'id already appears in its results.jsonl, and append new '
                        'results to the same files. Pass a directory path, or '
                        '"latest" to auto-pick the newest timestamp subdir under '
                        'output_dir. Implies --no_timestamp (writes into the '
                        'resumed dir, not a new timestamped one).')
    args = p.parse_args()

    overrides: Dict[str, Any] = {}
    for k in ['data_path', 'output_dir', 'model_name', 'api_key', 'base_url', 'max_turns']:
        v = getattr(args, k)
        if v is not None:
            overrides[k] = v
    if args.vision is not None:
        overrides['vision'] = args.vision

    cfg = load_config(args.config, overrides)
    if not cfg.get('data_path') or not cfg.get('model_name'):
        print('ERROR: --data_path and --model_name (or a config) are required.')
        sys.exit(1)

    # Archive each run under a timestamped subdirectory so reruns never
    # overwrite previous results/trajectories/workdirs. Override with
    # --no_timestamp to write directly into output_dir (legacy behavior).
    # Use Beijing time (Asia/Shanghai, UTC+8) regardless of the server's
    # system timezone — the box is on UTC and that would be 8h off.
    if ZoneInfo is not None:
        _now = datetime.now(ZoneInfo('Asia/Shanghai'))
    else:  # pragma: no cover — fallback for Python < 3.9
        _now = datetime.utcnow() + timedelta(hours=8)
    run_timestamp = _now.strftime('%Y%m%d_%H%M%S')

    # Resume: continue an existing run instead of starting a new timestamped
    # directory. Load already-finished task ids from the resumed results.jsonl
    # so they are skipped, and append new rows to the same files.
    resume_dir: Optional[str] = None
    done_ids: set = set()
    if args.resume:
        if args.resume == 'latest':
            base = cfg['output_dir']
            cands = [d for d in glob.glob(os.path.join(base, '*'))
                     if os.path.isdir(d) and re.match(r'^\d{8}_\d{6}$',
                                                      os.path.basename(d))]
            if not cands:
                print(f'ERROR: --resume latest found no timestamped subdir under {base}')
                sys.exit(1)
            resume_dir = max(cands)
        else:
            resume_dir = os.path.abspath(args.resume)
        res_file = os.path.join(resume_dir, 'results.jsonl')
        if not os.path.isfile(res_file):
            print(f'ERROR: --resume dir has no results.jsonl: {res_file}')
            sys.exit(1)
        for line in open(res_file, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            try:
                done_ids.add(str(json.loads(line).get('id')))
            except json.JSONDecodeError:
                continue
        cfg['output_dir'] = resume_dir
        print(f'Resuming into {resume_dir}; {len(done_ids)} task(s) already done '
              f'and will be skipped.')
    elif not args.no_timestamp:
        cfg['output_dir'] = os.path.join(cfg['output_dir'], run_timestamp)
    cfg['run_timestamp'] = run_timestamp

    os.makedirs(cfg['output_dir'], exist_ok=True)
    tasks = load_tasks(cfg['data_path'])
    # task_ids: --task_ids CLI arg takes precedence, else fall back to config.
    task_ids_str = args.task_ids or cfg.get('task_ids')
    if task_ids_str:
        if isinstance(task_ids_str, list):
            wanted = {str(x) for x in task_ids_str}
        else:
            wanted = {x.strip() for x in str(task_ids_str).split(',') if x.strip()}
        tasks = [t for t in tasks if _matches_task_id(t, wanted)]
        print(f'Filtered to {len(tasks)} task(s) matching task_ids={sorted(wanted)}')
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]

    print(f'Loaded {len(tasks)} tasks from {cfg["data_path"]}')
    print(f'Model: {cfg["model_name"]} @ {cfg["base_url"]}  vision={cfg.get("vision", False)}')

    # Output paths (stable across resume vs fresh run).
    out_results = os.path.join(cfg['output_dir'], 'results.jsonl')
    out_traj = os.path.join(cfg['output_dir'], 'trajectories.jsonl')

    bot = build_agent(cfg)
    results: List[Dict[str, Any]] = []
    correct = 0.0

    # On resume, seed results with the already-finished rows so the final
    # files stay complete and the accuracy covers all tasks.
    if args.resume:
        seen = set()
        for line in open(out_results, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get('id') in seen:
                continue
            seen.add(r.get('id'))
            results.append(r)
            correct += float(r.get('score') or (1.0 if r.get('correct') else 0.0))
        # trajectories are kept as-is on disk; we only append new ones below.

    def _write_summary() -> None:
        summary = {
            'n': len(results), 'correct': correct,
            'accuracy': correct / len(results) if results else 0.0,
            'model_name': cfg['model_name'],
            'data_path': cfg['data_path'],
            'vision': bool(cfg.get('vision', False)),
            'run_timestamp': cfg.get('run_timestamp', ''),
            'resumed': bool(args.resume),
        }
        with open(os.path.join(cfg['output_dir'], 'summary.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    for i, task in enumerate(tasks):
        if str(task['id']) in done_ids:
            print(f'\n[{i+1}/{len(tasks)}] task={task["id"]}  gt={task["answer"]}  '
                  f'SKIPPED (already done)')
            continue
        print(f'\n[{i+1}/{len(tasks)}] task={task["id"]}  gt={task["answer"]}')
        try:
            res = run_task(bot, task, cfg['max_turns'], cfg.get('vision', False))
        except Exception as e:
            traceback.print_exc()
            res = {'final_text': f'ERROR: {e}', 'predicted_answer': '',
                   'trajectory': []}

        sc = score_mcq(res['predicted_answer'], str(task['answer'] or ''))
        correct += sc['score']
        row = {
            'id': task['id'],
            'question': task['question'],
            'ground_truth': task['answer'],
            'predicted_answer': res['predicted_answer'],
            'pred_option': sc['pred'],
            'gt_option': sc['gt'],
            'correct': sc['correct'],
            'score': sc['score'],
            'final_text': res['final_text'],
            'trajectory': res.get('trajectory', []),
        }
        results.append(row)
        # Incremental append: a crash after this point still keeps every
        # finished task on disk.
        with open(out_results, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        with open(out_traj, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'id': task['id'],
                                'trajectory': row.get('trajectory', [])},
                               ensure_ascii=False) + '\n')
        done_ids.add(str(task['id']))
        print(f'  -> pred={sc["pred"]} gt={sc["gt"]} correct={sc["correct"]}')
        _write_summary()

    _write_summary()
    summary = json.load(open(os.path.join(cfg['output_dir'], 'summary.json')))
    print(f'\n=== Accuracy: {correct}/{len(results)} = {summary["accuracy"]:.2%} ===')
    print(f'Results: {out_results}')
    print(f'Trajectories: {out_traj}')


if __name__ == '__main__':
    main()
