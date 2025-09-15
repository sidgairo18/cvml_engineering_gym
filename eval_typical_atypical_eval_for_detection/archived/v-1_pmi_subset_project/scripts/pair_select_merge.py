import json, argparse, os, yaml
import pandas as pd

def load(path): 
    with open(path,'r') as f: return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", choices=["COCO","LVIS"], required=True)
    ap.add_argument("--pairs_cfg", default="augment_eval/configs/pairs_config.yaml")
    ap.add_argument("--indir", default="augment_eval/outputs/pairs")
    ap.add_argument("--out", default="augment_eval/outputs/pairs/final_pairs.json")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.pairs_cfg))
    base = load(os.path.join(args.indir, f"{args.ds}_pairs_selected.json"))
    if cfg["llm_merge"]["use_llm"]:
        llm = load(cfg["llm_merge"]["llm_pairs_json"])  # [{A,B,scene,relation,type,plausibility}, ...]
        # take intersection by names; union a few high-plausibility atypicals
        atyp_llm = [p for p in llm if p.get("type")=="atypical" and p.get("plausibility",0.0)>=0.6]
        # simple merge: keep PMI-typicals, PMI-atyp candidates, add LLM atypicals not already present
        names = {(r["A"], r["B"]) for r in base["typical"]} | {(r["A"], r["B"]) for r in base["atypical_candidates"]}
        merged_atyp = base["atypical_candidates"] + [p for p in atyp_llm if (p["A"],p["B"]) not in names]
        out = {"typical": base["typical"], "atypical": merged_atyp}
    else:
        out = {"typical": base["typical"], "atypical": base["atypical_candidates"]}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

if __name__=="__main__":
    main()

