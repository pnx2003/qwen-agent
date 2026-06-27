#!/usr/bin/env python3
# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
#
# Run a VQA multiple-choice benchmark with a vision-capable LLM via qwen_agent.
#
# Input format (jsonl, one dict per line), e.g. your dat_pat_*_vqa.jsonl:
#   {"query": "...", "images": ["/abs/path.png"], "answer": "A",
#    "vqa_type": "keyhole", "question_id": "dat_pat_q01", ...}
#
# Usage:
#   python3 run_vqa_mcq.py \
#       --data /cpfs01/pnx/AgentFlow/data/data_pat_q01_q15/dat_pat_q01_q15_vqa.jsonl \
#       --model openai/gpt-5.5 \
#       --api-base https://openai.sufy.com/v1 \
#       --api-key-env MY_API_KEY \
#       --output results.jsonl
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import List, Optional

# Make `qwen_agent` importable when running from a checkout that isn't pip-installed.
import os as _os
import sys as _sys
_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from qwen_agent.llm.schema import ContentItem, Message


def build_messages(query: str, image_paths: List[str], system_prompt: str) -> List[Message]:
    """Build a multimodal user message: images first, then the question text."""
    content: List[ContentItem] = []
    for p in image_paths:
        # qwenvl_oai accepts local paths directly (file:// or absolute) and
        # base64-encodes them internally via encode_image_as_base64.
        content.append(ContentItem(image=p))
    content.append(ContentItem(text=query))
    msgs = []
    if system_prompt:
        msgs.append(Message(role='system', content=system_prompt))
    msgs.append(Message(role='user', content=content))
    return msgs


# ---- MCQ option extraction (ported from AgentFlow's _extract_mcq_option) ----
def extract_mcq_option(text: str) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    lines = text.split('\n')

    # 1) "Answer: X" on its own line
    for line in reversed(lines):
        m = re.match(r'^\*{0,3}\s*[Aa]nswer\*{0,3}\s*[:：]\s*([A-Ja-j])\s*$', line.strip())
        if m:
            return m.group(1).upper()
    for line in reversed(lines):
        m = re.match(r'^\*{0,3}\s*[Aa]nswer\*{0,3}\s*[:：]\s*([A-Ja-j])', line.strip())
        if m:
            return m.group(1).upper()

    # 2) explicit markers
    marker_patterns = [
        r'(?i)(?:the\s+)?answer\s+is\s+([A-Ja-j])(?![a-zA-Z])',
        r'(?i)(?:choice|option|select|choose)\s+([A-Ja-j])(?![a-zA-Z])',
        r'(?i)(?:correct\s+)?answer\s*[:：]\s*([A-Ja-j])(?![a-zA-Z])',
    ]
    for p in marker_patterns:
        ms = re.findall(p, text)
        if ms:
            return ms[-1].upper()

    # 3) bold/markdown
    for p in [r'\*\*([A-Ja-j])\*\*', r'__([A-Ja-j])__', r'\*([A-Ja-j])\*']:
        ms = re.findall(p, text)
        if ms:
            return ms[-1].upper()

    # 4) bare letter
    m = re.match(r'^\s*([A-Ja-j])(?![a-zA-Z])\s*$', text)
    if m:
        return m.group(1).upper()

    # 5) last standalone letter near the end
    tail = ' '.join(lines[-5:])
    ms = re.findall(r'(?<![a-zA-Z])([A-Ja-j])(?![a-zA-Z])', tail)
    if ms:
        return ms[-1].upper()
    return None


SYSTEM_PROMPT = (
    'You are a test taker. Look at the image(s) and answer the multiple-choice question. '
    'After your reasoning, you MUST output your final answer on its own line in this exact format:\n\n'
    'Answer: X\n\n'
    'where X is the single uppercase option letter (e.g. A, B, C, D). '
    'Do not wrap it in markdown, do not output anything after that line.'
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Path to the VQA jsonl benchmark')
    ap.add_argument('--model', required=True, help='Model name as known to the API, e.g. openai/gpt-5.5')
    ap.add_argument('--api-base', required=True, help='OpenAI-compatible base_url')
    ap.add_argument('--api-key-env', default='OPENAI_API_KEY', help='Env var holding the API key')
    ap.add_argument('--output', default='vqa_mcq_results.jsonl', help='Where to write per-item results')
    ap.add_argument('--llm-type', default='qwenvl_oai',
                    help='qwen_agent llm backend (qwenvl_oai for vision via OpenAI-compatible API)')
    ap.add_argument('--max-tasks', type=int, default=None, help='Limit number of tasks (debug)')
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env, '')
    if not api_key:
        sys.exit(f'ERROR: env var {args.api_key_env} is not set')

    # Lazy import so --help works without deps installed.
    from qwen_agent.llm import get_chat_model

    llm = get_chat_model(
        {
            'model_type': args.llm_type,        # 'qwenvl_oai' = vision via OpenAI-compatible API
            'model': args.model,
            'model_server': args.api_base,      # oai.py also accepts 'api_base'/'base_url'
            'api_key': api_key,
            'generate_cfg': {'temperature': 0.0},
        }
    )

    # Load data
    items = []
    with open(args.data, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if args.max_tasks:
        items = items[:args.max_tasks]
    print(f'Loaded {len(items)} tasks from {args.data}', flush=True)

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

        msgs = build_messages(query, images, SYSTEM_PROMPT)
        try:
            resp = llm.chat(messages=msgs, stream=False)
            # BaseLLM.chat returns List[Message]; take last assistant content.
            text = ''
            for m in resp:
                if getattr(m, 'content', None):
                    text = m.content if isinstance(m.content, str) else str(m.content)
        except Exception as e:
            text = ''
            print(f'  [{qid}] LLM error: {e}', flush=True)

        pred = extract_mcq_option(text)
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
            'raw_response': (text or '')[:2000],
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
        fout.flush()
        print(f'  [{i}/{len(items)}] {qid} type={vqa_type} gt={gt} pred={pred} {"OK" if ok else "X"}', flush=True)

    fout.close()

    # ---- Summary ----
    total = len(items)
    print('\n========== Summary ==========')
    print(f'Total: {total}  Correct: {correct}  Accuracy: {correct/total:.4f}' if total else 'No tasks')
    print('\nBy vqa_type:')
    for t in sorted(by_type_total):
        c, n = by_type_correct[t], by_type_total[t]
        print(f'  {t:20s} {c}/{n}  acc={c/n:.4f}')

    summary = {
        'total': total,
        'correct': correct,
        'accuracy': correct / total if total else 0.0,
        'by_vqa_type': {t: {'correct': by_type_correct[t], 'total': by_type_total[t],
                            'accuracy': by_type_correct[t] / by_type_total[t]}
                        for t in sorted(by_type_total)},
    }
    summary_path = os.path.splitext(args.output)[0] + '_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'\nSummary written to {summary_path}')
    print(f'Per-item results written to {args.output}')


if __name__ == '__main__':
    main()
