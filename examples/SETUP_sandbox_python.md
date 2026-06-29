# 纯 Qwen-Agent 代码执行 Agent：从零上手教程（无 Docker，sandbox 式）

用 Qwen-Agent 框架搭一个「只挂一个 Python 代码执行工具」的 agent，在多模态 MCQ benchmark 上推理 + 评测。代码执行**不依赖 Docker**，而是用一个进程内 sandbox（受限 `__builtins__` + 导入白名单 + 超时 + per-task 工作目录）替代 Qwen-Agent 自带的 `code_interpreter`（它强依赖 Docker，服务器没 Docker 时用不了）。

这是 AgentFlow `configs/infer/mybenchmark/mybenchmark_tool_gpt5.5.json` 的 Qwen-Agent 版对应物：同样只挂一个 code 工具、同样读 AgentFlow 格式的 benchmark 数据、同样用 `my_mcq` 判分。

> 本教程假设你在 `/cpfs01/pnx` 下，Qwen-Agent 仓库在 `/cpfs01/pnx/Qwen-Agent`。

---

## 0. 整体架构

```
benchmark.jsonl (题目+图+答案)
        │
        ▼
benchmark_sandbox_python.py  (runner)
        │  逐题构造 [图, 题目] 的 user 消息
        ▼
FnCallAgent (Qwen-Agent 编排)
        │  LLM 生成 code → 调工具 → 结果回灌 → 再调 LLM … 直到给出 Answer: X
        ▼
sandbox_python 工具 (qwen_agent/tools/sandbox_python.py)
        │  files(图) 落盘到 per-task work_dir
        │  exec(code, 受限globals)  ← 进程内 sandbox，无 Docker
        │  出图以 ContentItem(image=) 回灌
        ▼
results.jsonl / trajectories.jsonl / summary.json  (my_mcq 判分)
```

**工具怎么拿到图**（关键，纯 Qwen-Agent 机制）：
工具继承 `BaseToolWithFileAccess` 且 `file_access=True`。框架自动把对话里出现过的图片/文件（`extract_files_from_messages(include_images=True)`）经 `files=` 参数传给工具，落盘到 per-task `work_dir`。模型用**文件名**读取：

```python
from PIL import Image
img = Image.open('dat_pat_q01.png')   # 文件就在工作目录里
```

不注入 `image_clue` 之类的变量，完全 Qwen-Agent 原生逻辑。

---

## 1. 相关文件

| 文件 | 作用 |
|---|---|
| `qwen_agent/tools/sandbox_python.py` | 自定义 code 工具 `sandbox_python`（无 Docker 的 sandbox 执行） |
| `examples/benchmark_sandbox_python.py` | benchmark 推理 + 评测 runner |
| `examples/mybenchmark_tool_gpt5.5_qwenagent.json` | 示例 config（仿 AgentFlow 格式） |
| `examples/README_sandbox_python.md` | 速查版说明 |
| `examples/SETUP_sandbox_python.md` | 本教程（从零上手） |

---

## 2. 环境准备（从零）

需要一个装了 **Qwen-Agent** 和**数据科学库**的 Python 环境。下面给两种方式。

### 方式 A：新建独立环境（推荐，干净）

```bash
# 1) 建环境（Python 3.10+）
conda create -n qwenagent python=3.12 -y
conda activate qwenagent

# 2) 安装 Qwen-Agent（从本仓库源码）+ 最小依赖
cd /cpfs01/pnx/Qwen-Agent
pip install -e .

# 3) 安装 code 工具预置的数据科学库
pip install numpy pandas scipy scikit-learn matplotlib seaborn opencv-python pillow
```

### 方式 B：复用现有 agentflow conda 环境

`/cpfs01/pnx/miniconda3/envs/agentflow` 已经有 openai / numpy / pandas / cv2 / matplotlib 等，只缺 qwen-agent 及其依赖，补装即可：

```bash
cd /cpfs01/pnx/Qwen-Agent

# 源码安装 qwen-agent 本体（不拉依赖，避免冲突）
conda run -n agentflow pip install -e . --no-deps

# 补 qwen-agent 运行所需依赖
conda run -n agentflow pip install json5 dashscope tiktoken jsonlines
```

### 验证安装

```bash
conda run -n agentflow python -c "from qwen_agent.tools.sandbox_python import SandboxPython; print('OK', SandboxPython({'root_work_dir':'/tmp/x'}).name)"
# 期望输出: OK sandbox_python
```

---

## 3. 工具自测（不调 LLM，确认 sandbox 能跑）

跑一段内联代码验证工具的执行、状态保持、出图、安全：

```bash
cd /cpfs01/pnx/Qwen-Agent
conda run -n agentflow python -c "
from qwen_agent.tools.sandbox_python import SandboxPython
t = SandboxPython({'root_work_dir':'/tmp/sb_smoke'})
img='/cpfs01/pnx/AgentFlow/data/data_pat_q01_q15/images/dat_pat_q01.png'
# 第1轮：用文件名读图 + 存一个变量
r1 = t.call({'code':'from PIL import Image\nimport numpy as np\nimg=Image.open(\"dat_pat_q01.png\")\nprint(\"size=\",img.size)\nsaved_var=7'}, files=[img], task_id='smoke')
print('TURN1:', r1[0].text)
# 第2轮：状态保持 + 出图
r2 = t.call({'code':'import matplotlib.pyplot as plt\nplt.plot([0,1,2],[0,1,4])\nplt.savefig(\"out.png\")\nprint(\"persisted=\",saved_var)'}, files=[img], task_id='smoke')
print('TURN2:', r2[0].text, '| n_images:', len(r2)-1)
# 第3轮：安全检查（subprocess 应被拦）
r3 = t.call({'code':'import subprocess'}, task_id='smoke')
print('TURN3 blocked:', 'blocked' in r3[0].text)
"
```

期望看到：
- `TURN1: size= (902, 171)` —— 模型用文件名读图成功
- `TURN2: persisted= 7 | n_images: 1` —— 状态跨轮保持 + 出图回传
- `TURN3 blocked: True` —— 沙箱安全生效

---

## 4. 输入数据格式（JSONL）

输入是 **JSONL** 文件：每行一个 JSON 对象，代表一道题。做了多别名兼容，可直接复用 AgentFlow 的数据。

### 4.1 支持的字段

| 用途 | 可用字段名（按优先级，任选其一） | 类型 |
|---|---|---|
| 任务 ID | `id` / `task_id` / `question_id` / `key_question` | string |
| 题目文本 | `query` / `question` / `input` / `prompt` | string |
| 标准答案 | `answer` / `ground_truth` / `expected` / `gt` | string（MCQ 为选项字母如 `"A"`） |
| 图片 | `images` / `image` / `image_byte` / `image_path` / `images_list` | string 或 list[string] |

图片值支持三种：**本地路径**、**URL**、**base64 字符串**。相对路径相对 jsonl 文件所在目录解析。其它字段（`vqa_type`、`question_type`、`data-source` 等）原样保留进 `metadata`，不影响推理。

### 4.2 JSONL 示例

多图 + 完整字段（AgentFlow 标准格式）：

```jsonl
{"query": "题目文本...哪个选项正确？", "images": ["/path/to/img.png"], "answer": "A", "task_id": "dat_pat_keyhole", "question_id": "dat_pat_q01", "question_type": "mcq"}
{"query": "题目文本...哪个选项正确？", "images": ["/path/to/img2.png"], "answer": "D", "task_id": "dat_pat_keyhole", "question_id": "dat_pat_q02", "question_type": "mcq"}
```

单图 + 单字段也行：

```jsonl
{"question": "...", "image": "/path/to/img.png", "answer": "B"}
```

### 4.3 准备你自己的数据

把你的题目写成上面的 JSONL 即可。若是已有的 AgentFlow jsonl（如 `/cpfs01/pnx/AgentFlow/data/data_pat_q01_q15/dat_pat_q01_q15_vqa.jsonl`），可直接用，无需改。

---

## 5. Config 格式（JSON）

Config 是一个 JSON 文件，字段尽量与 AgentFlow 对齐。示例见 `examples/mybenchmark_tool_gpt5.5_qwenagent.json`。

```json
{
  "benchmark_name": "dat_pat_q01_q15_vqa_tool",
  "model_name": "openai/gpt-5.4",
  "api_key": "sk-xxx",
  "base_url": "https://openai.sufy.com/v1",
  "max_turns": 40,
  "max_retries": 3,
  "sandbox_timeout": 300,
  "vision": false,
  "available_tools": ["sandbox_python"],
  "dataset_source": "local",
  "data_path": "/cpfs01/pnx/AgentFlow/data/data_pat_q01_q15/dat_pat_q01_q15_vqa.jsonl",
  "system_prompt": [
    "## Available Tools",
    "1. **sandbox_python** - Execute Python code in a sandboxed namespace",
    "   - Task images are in the working directory: open by filename,",
    "     e.g. `from PIL import Image; img = Image.open('dat_pat_q01.png')`",
    "   - Saved figures (plt.savefig) are returned to you",
    ""
  ],
  "evaluate_results": true,
  "evaluation_metric": "my_mcq",
  "output_dir": "infer_results/dat_pat_q01_q15_vqa/gpt5.5/qwen_agent_sandbox",
  "save_results": true,
  "save_trajectories": true
}
```

### Config 字段说明

| 字段 | 必填 | 说明 |
|---|---|---|
| `model_name` | 是 | 模型名（发给 OpenAI 兼容接口的 `model`） |
| `api_key` | 是 | API key |
| `base_url` | 是 | OpenAI 兼容接口地址（`.../v1`） |
| `data_path` | 是 | benchmark JSONL 路径 |
| `output_dir` | 是 | 结果输出目录 |
| `max_turns` | 否 | 每题最大对话轮数，默认 40 |
| `max_retries` | 否 | LLM 调用重试次数，默认 3 |
| `sandbox_timeout` | 否 | 单次代码执行超时秒数，默认 300 |
| `vision` | 否 | 模型端点是否支持图片输入。`true`：图也放进 user 消息给 VLM 看；`false`（默认）：图只放进 code 工作目录，模型写代码读取 |
| `system_prompt` | 否 | string 或 list[string]（每行一项）。会自动追加 `Answer: X` 格式约束 |
| `evaluation_metric` | 否 | 判分指标，目前支持 `my_mcq` |
| `max_input_tokens` | 否 | 传给 LLM 的 `max_input_tokens` |

> **关于 vision**：`openai.sufy.com` + `openai/gpt-5.4` 这个端点若支持发图就设 `true`（图同时给 VLM 看 + 给 code 工具读）；若不支持就保持 `false`（图只进 code 工作目录，模型写代码分析）。不确定就先 `false` 试。

---

## 6. 运行

所有命令都在 `/cpfs01/pnx/Qwen-Agent` 下执行。

### 6.1 先跑 1~2 道试通

```bash
cd /cpfs01/pnx/Qwen-Agent
conda run -n agentflow python examples/benchmark_sandbox_python.py \
    --config examples/mybenchmark_tool_gpt5.5_qwenagent.json \
    --max_tasks 2
```

确认：API 能调通、工具被调用、能提取出 `Answer: X`、判分正常。

### 6.2 全量跑

```bash
conda run -n agentflow python examples/benchmark_sandbox_python.py \
    --config examples/mybenchmark_tool_gpt5.5_qwenagent.json
```

### 6.3 复用 AgentFlow 原 config

工具会自动用 `sandbox_python`，无需改原 config：

```bash
conda run -n agentflow python examples/benchmark_sandbox_python.py \
    --config /cpfs01/pnx/AgentFlow/configs/infer/mybenchmark/mybenchmark_tool_gpt5.5.json \
    --max_tasks 2
```

### 6.4 命令行参数（覆盖 config）

```
--config            config 路径
--data_path         覆盖数据路径
--output_dir        覆盖输出目录
--model_name        覆盖模型名
--api_key           覆盖 api_key
--base_url          覆盖 base_url
--max_turns         覆盖最大轮数
--max_tasks N       只跑前 N 道
--task_ids a,b,c    只跑指定 ID
--vision / --no-vision   是否给模型发图（默认 no-vision）
```

例如换模型：

```bash
conda run -n agentflow python examples/benchmark_sandbox_python.py \
    --config examples/mybenchmark_tool_gpt5.5_qwenagent.json \
    --model_name your-model-name \
    --api_key sk-xxx \
    --base_url http://your-server/v1
```

---

## 7. 输出解读

在 `output_dir` 下生成：

| 文件 | 内容 |
|---|---|
| `results.jsonl` | 每题一行：`id`/`question`/`ground_truth`/`predicted_answer`/`pred_option`/`gt_option`/`correct`/`score`/`final_text` |
| `trajectories.jsonl` | 每题完整对话轨迹（含工具调用与结果） |
| `summary.json` | 汇总：`n`/`correct`/`accuracy`/`model_name`/`data_path` |
| `sandbox_workdir/task_<id>/` | 每题的 sandbox 工作目录（含落盘的题目图、生成的图等） |

`results.jsonl` 单行示例：

```json
{"id": "dat_pat_q01", "question": "...", "ground_truth": "A", "predicted_answer": "A", "pred_option": "A", "gt_option": "A", "correct": true, "score": 1.0, "final_text": "... Answer: A"}
```

终端最后会打印准确率：

```
=== Accuracy: 11/15 = 73.33% ===
```

---

## 8. 工具机制详解（sandbox_python）

完全遵循 Qwen-Agent 工具范式，5 个要点：

1. **文件传入**：`BaseToolWithFileAccess` + `file_access=True`。框架把对话里的图片/文件经 `files=` 传入，落盘到 per-task `work_dir`。模型用文件名读取（`Image.open('x.png')`、`pd.read_csv('x.csv')`）。
2. **执行**：进程内 `exec`，受限 `__builtins__` + 导入白名单（pandas/numpy/scipy/sklearn/matplotlib/seaborn/cv2/PIL 等），stdout/stderr 捕获，signal 超时。
3. **状态保持**：同一题多次调用复用同一 kernel，变量跨轮保持（像 Jupyter kernel）。
4. **出图回传**：`plt.savefig(...)` 或新写入的图片文件，以 `List[ContentItem(image=path)]` 返回，框架作为多模态结果回灌给模型。
5. **安全**：`subprocess` 等非白名单模块被拒；每题独立工作目录互不干扰。

---

## 9. 评测（my_mcq）

从模型最终回复里提取选项字母（支持 `Answer: X`、`**X**`、`the answer is X` 等多种格式），与标准答案比对，正确得 1.0。

---

## 10. 常见问题

- **PIL / cv2 报 `ImportError: Security: Import of module 'xxx' is blocked.`**
  需要导入的库不在白名单。编辑 `qwen_agent/tools/sandbox_python.py` 的 `ALLOWED_MODULES`，加入该模块的根名（如 `'PIL'`）即可。

- **模型不调工具直接答 / 答非所问**
  - 确认 `system_prompt` 里说明了工具用途；
  - `max_turns` 给够（默认 40）；
  - 若端点不支持 vision 又必须看图，模型只能靠写代码读图分析，`system_prompt` 里强调「先写代码分析图片再作答」。

- **端点不支持发图（vision）**
  保持 `--no-vision`（默认），图只进 code 工作目录，模型写代码分析。

- **`ModuleNotFoundError: No module named 'qwen_agent'` / `'json5'`**
  环境没装好，回到第 2 步重装。复用 agentflow env 时记得补装 `json5 dashscope tiktoken jsonlines`。

- **超时**
  调大 `sandbox_timeout`（config 字段）或 `--max_turns`。

---

## 11. 快速命令汇总

```bash
# 进目录
cd /cpfs01/pnx/Qwen-Agent

# (一次性) 复用 agentflow env 装依赖
conda run -n agentflow pip install -e . --no-deps
conda run -n agentflow pip install json5 dashscope tiktoken jsonlines

# 工具自测
conda run -n agentflow python -c "from qwen_agent.tools.sandbox_python import SandboxPython; print('OK')"

# 试跑 2 道
conda run -n agentflow python examples/benchmark_sandbox_python.py \
    --config examples/mybenchmark_tool_gpt5.5_qwenagent.json --max_tasks 2

# 全量跑
conda run -n agentflow python examples/benchmark_sandbox_python.py \
    --config examples/mybenchmark_tool_gpt5.5_qwenagent.json
```
