#!/usr/bin/env python3
import argparse, os, json
import pandas as pd
from collections import defaultdict

def _load_llm_pairs(path):
    if not path or not os.path.exists(path):
        return []
    with open(path,'r') as f:
        data = json.load(f)
    return data

def _plausibility_lookup(llm_pairs):
    table = {}
    for p in llm_pairs:
        key = (p['a'], p['b'], p.get('type','typical'))
        table[key] = {'plausibility': float(p.get('plausibility', 1.0)),
                      'scene': p.get('scene'),
                      'relation': p.get('relation')}
    return table

def _rank_partners(df, anchor):
    sel_rows = []
    for _,row in df.iterrows():
        a = row['cat_a_name']; b = row['cat_b_name']
        if a == anchor or b == anchor:
            partner = b if a == anchor else a
            sel_rows.append(dict(anchor=anchor, partner=partner, pmi=float(row['pmi']),
                                 count_a=int(row['count_a']), count_b=int(row['count_b']),
                                 count_ab=int(row['count_ab']), pair=(a,b)))
    if not sel_rows:
        return pd.DataFrame([]), pd.DataFrame([])
    sub = pd.DataFrame(sel_rows)
    typical = sub.sort_values('pmi', ascending=False).reset_index(drop=True)
    atypical = sub.sort_values('pmi', ascending=True).reset_index(drop=True)
    return typical, atypical

def _choose_k_anchors(df, k):
    counts = defaultdict(int)
    for _,row in df.iterrows():
        counts[row['cat_a_name']] = max(counts[row['cat_a_name']], int(row['count_a']))
        counts[row['cat_b_name']] = max(counts[row['cat_b_name']], int(row['count_b']))
    anchors = sorted(counts.keys(), key=lambda c: counts[c], reverse=True)[:k]
    return anchors

def build_matched_pairs(df, k, pairs_per_anchor_typ=1, pairs_per_anchor_aty=1, min_plaus=0.0, llm_pairs=None):
    llm_table = _plausibility_lookup(llm_pairs or [])
    anchors = _choose_k_anchors(df, k)
    matched = []
    flat_pairs = []
    match_id = 1
    for anchor in anchors:
        typical, atypical = _rank_partners(df, anchor)
        if typical.empty or atypical.empty:
            continue
        def is_plaus(a,b,typetag):
            key = (a,b,typetag)
            key2 = (b,a,typetag)
            if key in llm_table:
                return llm_table[key]['plausibility'] >= min_plaus
            if key2 in llm_table:
                return llm_table[key2]['plausibility'] >= min_plaus
            return True
        picked_typ = []
        for _,row in typical.iterrows():
            a,b = row['pair']; partner = row['partner']
            if not is_plaus(a,b,'typical'): continue
            picked_typ.append(dict(a=anchor, b=partner, pmi=float(row['pmi']), type='typical'))
            if len(picked_typ) >= pairs_per_anchor_typ: break
        picked_aty = []
        typ_partners = {p['b'] if p['a']==anchor else p['a'] for p in picked_typ}
        for _,row in atypical.iterrows():
            a,b = row['pair']; partner = row['partner']
            if partner in typ_partners: continue
            if not is_plaus(a,b,'atypical'): continue
            picked_aty.append(dict(a=anchor, b=partner, pmi=float(row['pmi']), type='atypical'))
            if len(picked_aty) >= pairs_per_anchor_aty: break
        if not picked_typ or not picked_aty: continue
        matched.append({'match_id': match_id, 'anchor': anchor, 'typical': picked_typ, 'atypical': picked_aty})
        for p in picked_typ: flat_pairs.append({**p, 'match_id': match_id, 'relation':'next to','scene':'street'})
        for p in picked_aty: flat_pairs.append({**p, 'match_id': match_id, 'relation':'on','scene':'indoor room'})
        match_id += 1
    return matched, flat_pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pmi_csv', required=True)
    ap.add_argument('--dataset', required=True, choices=['coco','lvis'])
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--pairs_per_anchor_typ', type=int, default=1)
    ap.add_argument('--pairs_per_anchor_aty', type=int, default=1)
    ap.add_argument('--llm_pairs_json', default=None)
    ap.add_argument('--min_plausibility', type=float, default=0.0)
    ap.add_argument('--out_matched_json', required=True)
    ap.add_argument('--out_pairs_flat_json', required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.pmi_csv)
    llm_pairs = _load_llm_pairs(args.llm_pairs_json)
    matched, flat = build_matched_pairs(df, args.k, args.pairs_per_anchor_typ, args.pairs_per_anchor_aty, args.min_plausibility, llm_pairs)

    os.makedirs(os.path.dirname(args.out_matched_json), exist_ok=True)
    with open(args.out_matched_json,'w') as f:
        json.dump({'dataset': args.dataset, 'matched': matched}, f, indent=2)
    with open(args.out_pairs_flat_json,'w') as f:
        json.dump({'dataset': args.dataset, 'pairs': flat}, f, indent=2)
    print(f"[OK] Wrote {args.out_matched_json} and {args.out_pairs_flat_json}")

if __name__ == '__main__':
    main()
