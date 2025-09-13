#!/usr/bin/env python3
import argparse, json, os
import pandas as pd

def load_llm_pairs(path):
    if not path or not os.path.exists(path):
        return []
    with open(path,'r') as f:
        data = json.load(f)
    # expected format: list of dicts: {"a":"person","b":"bicycle","type":"typical|atypical","plausibility":0.0-1.0,"scene":"street, daytime","relation":"on"}
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pmi_csv', required=True)
    ap.add_argument('--cat_map_json', required=True)
    ap.add_argument('--dataset', required=True, choices=['coco','lvis'])
    ap.add_argument('--top_typical', type=int, default=20)
    ap.add_argument('--top_atypical', type=int, default=20)
    ap.add_argument('--llm_pairs_json', default=None, help='optional LLM-curated pairs json')
    ap.add_argument('--out_json', required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.pmi_csv)
    with open(args.cat_map_json,'r') as f: cat_map = {int(k):v for k,v in json.load(f).items()}

    # typical by PMI rank
    typical_df = df.sort_values('pmi', ascending=False).head(args.top_typical)
    typical = [
        dict(a=row['cat_a_name'], b=row['cat_b_name'], type='typical', source='pmi', pmi=float(row['pmi']))
        for _,row in typical_df.iterrows()
    ]

    # atypical by low PMI (but keep some plausibility via simple heuristics: avoid identical super-categories if available)
    atypical_df = df.sort_values('pmi', ascending=True).head(args.top_atypical*3)  # sample extra, will filter
    atypical = []
    for _,row in atypical_df.iterrows():
        a, b = row['cat_a_name'], row['cat_b_name']
        # trivial plausibility filter (extend as needed)
        if any(x in a for x in ['background','other']) or any(x in b for x in ['background','other']):
            continue
        atypical.append(dict(a=a, b=b, type='atypical', source='pmi', pmi=float(row['pmi'])))
        if len(atypical) >= args.top_atypical:
            break

    # merge LLM proposals if provided (deduplicate by pair+type)
    llm_pairs = load_llm_pairs(args.llm_pairs_json)
    all_pairs = {(p['a'], p['b'], p['type']): p for p in (typical + atypical)}
    for p in llm_pairs:
        key = (p['a'], p['b'], p.get('type','typical'))
        all_pairs[key] = {**p, **dict(source='llm')}
    final = list(all_pairs.values())

    # attach default scenes/relations (can be edited later)
    for p in final:
        if 'scene' not in p:
            p['scene'] = 'street' if p['type']=='typical' else 'indoor room'
        if 'relation' not in p:
            p['relation'] = 'next to' if p['type']=='typical' else 'on'

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json,'w') as f:
        json.dump(dict(dataset=args.dataset, pairs=final), f, indent=2)
    print(f"[OK] Selected pairs saved to {args.out_json} (N={len(final)})")

if __name__ == '__main__':
    main()
