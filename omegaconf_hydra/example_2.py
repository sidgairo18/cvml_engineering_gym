from omegaconf import OmegaConf

cfg = OmegaConf.load("./configs/config1.yaml")
print(cfg.trainer.epochs)   # 10
