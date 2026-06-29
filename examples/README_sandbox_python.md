# 纯 Qwen-Agent 代码执行 Agent（无 Docker，sandbox 式）

用 Qwen-Agent 框架搭一个「只挂一个 Python 代码执行工具」的 agent，在多模态 MCQ benchmark 上推理评测。代码执行不依赖 Docker，而是用一个进程内 sandbox（受限 `__builtins__` + 导入白名单 + 超时 + per-task 工作目录）替代 `code_interpreter` 的 Docker 容器。

本目录是 AgentFlow `configs/infer/mybenchmark/mybenchmark_tool_gpt5.5.json` 的 Qwen-Agent 版对应物：同样只挂一个 code 工具、同样读 AgentFlow 格式的 benchmark 数据、同样用 `my_mcq` 判分。

## 文件

| 文件 | 作用 |
|---|---|
| `qwen_agent/tools/sandbox_python.py` | 自定义 code 工具 `sandbox_python`（无 Docker 的 sandbox 执行） |
| `examples/benchmark_sandbox_python.py` | benchmark 推理 + 评测 runner |
| `examples/mybenchmark_tool_gpt5.5_qwenagent.json` | 示例 config（仿 AgentFlow 格式） |
| `examples/README_sandbox_python.md` | 本文档 |

## 安装

需要一个装了 Qwen-Agent 和数据科学库的 Python 环境。

```bash
# 1. 安装 Qwen-Agent（本仓库源码安装）+ 最小依赖
cd /cpfs01/pnx/Qwen-Agent
pip install -e .

# 2. 代码执行用到的库（sandbox 里预置）
pip install numpy pandas scipy scikit-learn matplotlib seaborn opencv-python pillow

# 3. 若复用 conda 环境 agentflow（已含 openai / ds 库），补装 qwen-agent 依赖：
#    conda run -n agentflow pip install -e /cpfs01/pnx/Qwen-Agent --no-deps
#    conda run -n agentflow pip install json5 dashscope tiktoken jsonlines
```

## 输入格式

输入是 **JSONL**（每行一个 JSON 对象，一道题）。也兼容 `query`/`question`/`input`/`prompt` 等多种字段名，方便直接复用 AgentFlow 的数据。

### 支持的字段

| 字段（按优先级，任选其一） | 类型 | 说明 |
|---|---|---|
| `id` / `task_id` / `question_id` / `key_question` | string | 任务唯一 ID |
| `query` / `question` / `input` / `prompt` | string | 题目文本 |
| `answer` / `ground_truth` / `expected` / `gt` | string | 标准答案（MCQ 为选项字母，如 `"A"`） |
| `images` / `image` / `image_byte` / `image_path` / `images_list` | string 或 list[string] | 题目图片：本地路径、URL、或 base64 |

其它字段（如 `vqa_type`、`question_type`、`data-source` 等）会原样保留进 `metadata`，不参与推理。

### JSONL 示例

```jsonl
{"query": "题目文本...哪个选项正确？", "images": ["/path/to/img.png"], "answer": "A", "task_id": "dat_pat_keyhole", "question_id": "dat_pat_q01", "question_type": "mcq"}
{"query": "题目文本...哪个选项正确？", "images": ["/path/to/img2.png"], "answer": "D", "task_id": "dat_pat_keyhole", "question_id": "dat_pat_q02", "question_type": "mcq"}
```

单字段图片也支持：

```jsonl
{"question": "...", "image": "/path/to/img.png", "answer": "B"}
```

图片路径是相对路径时，会相对 jsonl 文件所在目录解析。

## Config 格式

Config 是一个 JSON 文件，字段尽量与 AgentFlow 对齐。

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
  "system_prompt": ["## Available Tools", "1. **sandbox_python** - ...", ""],
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

## 运行

```bash
cd /cpfs01/pnx/Qwen-Agent

# 用示例 config 全量跑
python examples/benchmark_sandbox_python.py \
    --config examples/mybenchmark_tool_gpt5.5_qwenagent.json

# 或直接复用 AgentFlow 原 config（工具会自动用 sandbox_python）
python examples/benchmark_sandbox_python.py \
    --config /cpfs01/pnx/AgentFlow/configs/infer/mybenchmark/mybenchmark_tool_gpt5.5.json

# 先跑 2 道试通
python examples/benchmark_sandbox_python.py \
    --config examples/mybenchmark_tool_gpt5.5_qwenagent.json \
    --max_tasks 2

# 若端点支持发图（vision），加 --vision
python examples/benchmark_sandbox_python.py \
    --config examples/mybenchmark_tool_gpt5.5_qwenagent.json \
    --vision
```

命令行参数会覆盖 config 里的同名字段：

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
--vision / --no-vision   是否给模型发图
```

## 输出

在 `output_dir` 下生成：

| 文件 | 内容 |
|---|---|
| `results.jsonl` | 每题一行：`id`/`question`/`ground_truth`/`predicted_answer`/`pred_option`/`gt_option`/`correct`/`score`/`final_text` |
| `trajectories.jsonl` | 每题完整对话轨迹（含工具调用与结果） |
| `summary.json` | 汇总：`n`/`correct`/`accuracy`/`model_name`/`data_path` |
| `sandbox_workdir/task_<id>/` | 每题的 sandbox 工作目录（含落盘的题目图、生成的图等） |

终端最后会打印：

```
=== Accuracy: 11/15 = 73.33% ===
```

## 工具机制（sandbox_python 怎么工作）

完全遵循 Qwen-Agent 的工具范式：

1. **文件传入**：继承 `BaseToolWithFileAccess`，`file_access=True`。框架自动把对话里出现的图片/文件（`extract_files_from_messages(include_images=True)`）经 `files=` 传给工具，落盘到 per-task `work_dir`。模型用**文件名**读取：
   ```python
   from PIL import Image
   img = Image.open('dat_pat_q01.png')   # 文件就在工作目录里
   ```
2. **执行**：进程内 `exec`，受限 `__builtins__` + 导入白名单（pandas/numpy/scipy/sklearn/matplotlib/seaborn/cv2/PIL 等），stdout/stderr 捕获，signal 超时。
3. **状态保持**：同一题多次调用复用同一 kernel，变量跨轮保持（像 Jupyter kernel）。
4. **出图回传**：`plt.savefig(...)` 或新写入的图片文件，以 `List[ContentItem(image=path)]` 返回，框架作为多模态结果回灌给模型。
5. **安全**：`subprocess` 等非白名单模块被拒；每题独立工作目录互不干扰。

## 评测（my_mcq）

从模型最终回复里提取选项字母（支持 `Answer: X`、`**X**`、`the answer is X` 等多种格式），与标准答案比对，正确得 1.0。

## 常见问题

- **PIL / cv2 报 ImportError: Security: blocked**：需要导入的库不在白名单。编辑 `sandbox_python.py` 的 `ALLOWED_MODULES` 加入即可。
- **模型不调工具直接答**：确认 `system_prompt` 里说明了工具用途；`max_turns` 给够。
- **端点不支持发图**：保持 `--no-vision`（默认），图只进 code 工作目录，模型写代码分析。
