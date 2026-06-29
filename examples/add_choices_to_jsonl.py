"""Add text `choices` to the angle-ranking questions (Q11-15) of the DAT PAT
benchmark jsonl, sourced from https://www.dat-prep.com/dat-pat-sample-question.

Only PART 3 (angle ranking, Q11-15) has text options (sort orders like
"4-1-3-2"); the other parts' options are images embedded in the question
figure and cannot be text-extracted, so they are left unchanged.

Usage:
    python examples/add_choices_to_jsonl.py \
        --in  /cpfs01/pnx/AgentFlow/data/data_pat_q01_q15/dat_pat_q01_q15_vqa.jsonl \
        --out /cpfs01/pnx/AgentFlow/data/data_pat_q01_q15/dat_pat_q01_q15_vqa_with_choices.jsonl
"""
import argparse
import json

# Options for the angle-ranking questions, keyed by source_question_number.
# Each value maps option letter -> ordering of the four angles (1-4, left to
# right in the figure), from smallest to largest interior angle.
ANGLE_CHOICES = {
    11: {'A': '4-1-3-2', 'B': '2-3-1-4', 'C': '1-2-4-3', 'D': '2-4-3-1'},
    12: {'A': '1-2-4-3', 'B': '3-2-4-1', 'C': '4-3-1-2', 'D': '1-4-3-2'},
    13: {'A': '2-4-3-1', 'B': '3-4-2-1', 'C': '3-1-4-2', 'D': '1-4-3-2'},
    14: {'A': '2-1-4-3', 'B': '3-4-2-1', 'C': '4-2-1-3', 'D': '3-2-4-1'},
    15: {'A': '2-1-3-4', 'B': '1-3-4-2', 'C': '3-4-1-2', 'D': '2-4-3-1'},
}


def format_choices(choices: dict) -> str:
    """Render choices as a human-readable options block."""
    return 'Options:\n' + '\n'.join(f'  {k}. {v}' for k, v in choices.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', dest='out', required=True)
    args = ap.parse_args()

    n_added = 0
    with open(args.inp, 'r', encoding='utf-8') as fin, \
         open(args.out, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            qnum = item.get('source_question_number')
            if qnum in ANGLE_CHOICES:
                choices = ANGLE_CHOICES[qnum]
                item['choices'] = choices
                # Also append the options to the query text so models that only
                # read `query` still see them.
                opts_text = format_choices(choices)
                if 'Options:' not in (item.get('query') or ''):
                    item['query'] = (item.get('query') or '').rstrip() + '\n\n' + opts_text
                n_added += 1
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f'Wrote {args.out}; added choices to {n_added} angle-ranking items.')


if __name__ == '__main__':
    main()
