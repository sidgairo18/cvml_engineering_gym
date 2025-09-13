import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()

"""
Run (basic): python example_3_hydra.py
Run: python example_3_hydra.py model=vit trainer.epochs=20
"""
