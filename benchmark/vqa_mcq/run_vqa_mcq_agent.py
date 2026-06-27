#!/usr/bin/env python3
# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
#
# Run a VQA multiple-choice benchmark with a tool-augmented multimodal agent.
#
# Unlike run_vqa_mcq.py (pure single-turn VQA), this uses qwen_agent's FnCallAgent
# with a python_executor tool so the model can write & run code to reason about the
# image (e.g. analyze pixels, count, measure). The FULL trajectory is saved:
# every LLM turn, every tool call + tool result, and the final answer.
#
# Input format (jsonl), e.g. dat_pat_*_vqa.jsonl:
#   {"query": "...", "images": ["/abs/path.png"], "answer": "A",
#    "vqa_type": "keyhole", "question_id": "dat_pat_q01", ...}
#
# Usage:
#   MY_API_KEY=... python3 run_vqa_mcq_agent.py \
#       --data /cpfs01/pnx/AgentFlow/data/data_pat_q01_q15/dat_pat_q01_q15_vqa.jsonl \
#       --model openai/gpt-5.5 \
#       --api-base https://openai.sufy.com/v1 \
#       --api-key-env MY_API_KEY \
#       --output agent_results.jsonl
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import List, Optional

# Make `qwen_agent` importable when running from a non-pip-installed checkout.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qwen_agent.agents import FnCallAgent
from qwen_agent.llm.schema import ContentItem, Message, FUNCTION, ROLE
from qwen_agent.tools import PythonExecutor


SYSTEM_PROMPT = (
    'You are a test taker answering a multiple-choice visual question. '
    'You are given the question image(s) and the question text. '
    'You have a python code interpreter tool you may use to help reason about the image '
    '(e.g. load the image with PIL, inspect regions, measure angles, count features). '
    'Use the tool only if it helps; otherwise reason directly. '
    'When you are sure, output your final answer on its own line in this EXACT format:\n\n'
    'Answer: X\n\n'
    'where X is the single uppercase option letter (e.g. A, B, C, D). '
    'Do not wrap it in markdown, do not output anything after that line.'
)


def build_messages(query: str, image_paths: List[str]) -> List[Message]:
    content: List[ContentItem] = [ContentItem(image=p) for p in image_paths]
    content.append(ContentItem(text=query))
    return [Message(role='user', content=content)]


def extract_mcq_option(text: str) -> Optional[str]:
    """Same robust A-J extraction as run_vqa_mcq.py / AgentFlow."""
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
    for p in [r'(?i)(?:the\s+)?answer\s+is\s+([A-Ja-j])(?![a-zA-Z])',
              r'(?i)(?:choice|option|select|choose)\s+([A-Ja-j])(?![a-zA-Z])',
              r'(?i)(?:correct\s+)?answer\s*[:：]\s*([A-Ja-j])(?![a-zA-Z])']:
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


def trajectory_to_dicts(messages: List[Message]) -> List[dict]:
    """Serialize the full agent trajectory (messages) to plain JSON-safe dicts.

    Uses pydantic model_dump() so FunctionCall/ContentItem become plain dicts,
    then strips out bulky base64 image data to keep the trajectory readable.
    """
    out = []
    for m in messages:
        if isinstance(m, dict):
            d = dict(m)
        else:
            d = m.model_dump()
        # Drop the raw image payload (could be a huge base64 string) -> keep a placeholder.
        content = d.get('content')
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get('image'):
                    item['image'] = '<image omitted>'
        # reasoning_content may also carry images
        rc = d.get('reasoning_content')
        if isinstance(rc, list):
            for item in rc:
                if isinstance(item, dict) and item.get('image'):
                    item['image'] = '<image omitted>'
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--api-base', required=True)
    ap.add_argument('--api-key-env', default='OPENAI_API_KEY')
    ap.add_argument('--output', default='vqa_agent_results.jsonl')
    ap.add_argument('--max-tasks', type=int, default=None)
    ap.add_argument('--max-turns', type=int, default=20, help='Max LLM calls per task (agent loop cap)')
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env, '')
    if not api_key:
        sys.exit(f'ERROR: env var {args.api_key_env} not set')

    llm_cfg = {
        'model_type': 'qwenvl_oai',
        'model': args.model,
        'model_server': args.api_base,
        'api_key': api_key,
        'generate_cfg': {'temperature': 0.0},
    }

    # Build the agent with a python_executor tool. PythonExecutor is not registered
    # by default (unsafe), so we pass the instantiated object directly.
    agent = FnCallAgent(
        llm=llm_cfg,
        function_list=[PythonExecutor(cfg={'get_answer_from_stdout': True, 'timeout_length': 20})],
        system_message=SYSTEM_PROMPT,
    )

    items = []
    with open(args.data, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if args.max_tasks:
        items = items[:args.max_tasks]
    print(f'Loaded {len(items)} tasks', flush=True)

    # Raise the agent's internal max-LLM-calls cap. fncall_agent binds this name
    # by value at import time, so patch it on the fncall_agent module itself.
    import qwen_agent.agents.fncall_agent as _fa
    _fa.MAX_LLM_CALL_PER_RUN = args.max_turns

    fout = open(args.output, 'w', encoding='utf-8')
    correct = 0
    by_type_correct = defaultdict(int)
    by_type_total = defaultdict(int)

    for i, item in enumerate(items, 1):
        qid = item.get('question_id') or item.get('id') or str(i)
        query = item['query']
        images = item.get('images', [])
        gt = str(item['answer']).strip().upper()
        vqa_type = item.get('vqa_type', 'unknown')

        msgs = build_messages(query, images)
        full_trajectory: List[Message] = []
        n_tool_calls = 0
        try:
            for partial in agent.run(msgs):
                # `run` yields the accumulated response so far; keep the longest.
                if partial:
                    full_trajectory = partial
            # Also get the final nonstream result to be safe.
            final = agent.run_nonstream(msgs)
            if final:
                full_trajectory = final
        except Exception as e:
            print(f'  [{qid}] agent error: {e}', flush=True)

        # Count tool calls & extract final assistant text from the trajectory.
        final_text = ''
        for m in full_trajectory:
            role = getattr(m, 'role', None)
            if role == FUNCTION or (isinstance(m, dict) and m.get('role') == FUNCTION):
                n_tool_calls += 1
            if role == 'assistant' or (isinstance(m, dict) and m.get('role') == 'assistant'):
                c = getattr(m, 'content', None) if not isinstance(m, dict) else m.get('content')
                if isinstance(c, str) and c:
                    final_text = c
        pred = extract_mcq_option(final_text)
        ok = (pred == gt) if pred else False
        if ok:
            correct += 1
            by_type_correct[vqa_type] += 1
        by_type_total[vqa_type] += 1

        rec = {
            'question_id': qid,
            'vqa_type': vqa_type,
            'ground_truth': gt,
            'predicted': pred,
            'correct': ok,
            'n_tool_calls': n_tool_calls,
            'final_text': (final_text or '')[:3000],
            'trajectory': trajectory_to_dicts(full_trajectory),
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
        fout.flush()
        print(f'  [{i}/{len(items)}] {qid} type={vqa_type} gt={gt} pred={pred} '
              f'tools={n_tool_calls} {"OK" if ok else "X"}', flush=True)

    fout.close()

    total = len(items)
    print('\n========== Summary ==========')
    print(f'Total: {total}  Correct: {correct}  Accuracy: {correct/total:.4f}' if total else 'No tasks')
    print('By vqa_type:')
    for t in sorted(by_type_total):
        c, n = by_type_correct[t], by_type_total[t]
        print(f'  {t:20s} {c}/{n}  acc={c/n:.4f}')
    summary = {
        'total': total, 'correct': correct,
        'accuracy': correct / total if total else 0.0,
        'by_vqa_type': {t: {'correct': by_type_correct[t], 'total': by_type_total[t],
                            'accuracy': by_type_correct[t] / by_type_total[t]}
                        for t in sorted(by_type_total)},
    }
    sp = os.path.splitext(args.output)[0] + '_summary.json'
    with open(sp, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'Summary -> {sp}\nResults -> {args.output}')


if __name__ == '__main__':
    main()
