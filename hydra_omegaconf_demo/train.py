import os
import time
import json
import random
from dataclasses import dataclass

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd


# (Optional) structured config for reference
@dataclass
class TrainerCfg:
    epochs: int = 3
    batch_size: int = 32
    lr: float = 0.001


def set_seed(seed: int):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def fake_train_step(step: int, cfg: DictConfig) -> float:
    # pretend to compute a loss that decreases a bit each step with noise
    base = 1.0 / (step + 1)
    noise = 0.05 * random.random()
    return base + noise


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # 1) Show the final composed config (after defaults + overrides)
    print("=== Effective Config ===")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # 2) Hydra diagnostics (cwd, output dir)
    print("\n=== Hydra diagnostics ===")
    print("original cwd           :", get_original_cwd())
    print("runtime.cwd            :", HydraConfig.get().runtime.cwd)
    print("runtime.output_dir     :", HydraConfig.get().runtime.output_dir)
    print("os.getcwd() (run dir)  :", os.getcwd())

    # 3) Reproducibility
    set_seed(cfg.seed)

    # 4) Tiny fake training loop
    history = []
    for epoch in range(cfg.trainer.epochs):
        start = time.time()
        loss = 0.0
        for step in range(10):
            loss += fake_train_step(step + epoch * 10, cfg)
        loss /= 10.0
        dur = time.time() - start
        print(f"Epoch {epoch+1}/{cfg.trainer.epochs} - loss={loss:.4f} - {dur*1000:.1f} ms")
        history.append({"epoch": epoch + 1, "loss": float(loss)})

    # 5) Save artifacts into the Hydra run directory (bullet-proof)
    run_dir = HydraConfig.get().runtime.output_dir  # absolute path to this run
    artifacts_dir = os.path.join(run_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    with open(os.path.join(artifacts_dir, "metrics.json"), "w") as f:
        json.dump({"final_loss": history[-1]["loss"], "history": history}, f, indent=2)

    with open(os.path.join(artifacts_dir, "config_dump.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    print("\nSaved: artifacts/metrics.json and artifacts/config_dump.yaml")


if __name__ == "__main__":
    main()

