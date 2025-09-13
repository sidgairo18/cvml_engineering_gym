#!/usr/bin/env python3
import argparse, os, json, random

TYPICAL_TEMPLATES = [
    "A high-quality photo of a {A} {rel} a {B} in a {scene}, realistic lighting, detailed textures, 35mm, daytime",
    "A {A} {rel} a {B} in a {scene}, natural colors, documentary style"
]
ATYPICAL_TEMPLATES = [
    "A realistic photo of a {A} {rel} a {B} in a {scene}, physically plausible composition, 35mm",
    "A {A} {rel} a {B} in a {scene}, consistent shadows and perspective, high detail"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['coco','lvis'])
    ap.add_argument('--pairs_json', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--per_pair_per_type', type=int, default=200)
    args = ap.parse_args()

    with open(args.pairs_json, 'r') as f:
        pairs_data = json.load(f)
    pairs = pairs_data['pairs']

    os.makedirs(args.out_dir, exist_ok=True)
    prompts_path = os.path.join(args.out_dir, 'prompts.jsonl')
    manifest_path = os.path.join(args.out_dir, 'manifest.jsonl')

    with open(prompts_path, 'w') as fpr, open(manifest_path, 'w') as fmf:
        for p in pairs:
            a, b = p['a'], p['b']
            rel = p.get('relation','next to')
            scene = p.get('scene','street')
            typ = p.get('type','typical')
            templates = TYPICAL_TEMPLATES if typ=='typical' else ATYPICAL_TEMPLATES
            for i in range(args.per_pair_per_type):
                tmpl = random.choice(templates)
                prompt = tmpl.format(A=a, B=b, rel=rel, scene=scene)
                entry = {'pair': {'a':a,'b':b,'type':typ,'relation':rel,'scene':scene,'match_id':p.get('match_id')},
                         'prompt': prompt, 'seed': random.randint(0,10**9),
                         'target_size': [800,800], 'need_masks': True}
                fpr.write(json.dumps(entry)+'\n')
                fmf.write(json.dumps({'prompt_id': i, 'status':'PENDING', 'pair':entry['pair']})+'\n')

    print(f"[OK] Prompts -> {prompts_path}")
    print(f"[OK] Manifest -> {manifest_path}")

if __name__ == '__main__':
    main()
