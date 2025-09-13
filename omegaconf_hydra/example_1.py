from omegaconf import OmegaConf

cfg = OmegaConf.create({
    "model": {"name": "resnet50", "lr": 0.001},
    "trainer": {"epochs": 10, "batch_size": 32}
})

print(cfg.model.name)       # resnet50
print(cfg.trainer.epochs)
#print(cfg["trainer"]["lr"]) # KeyError (doesn't exist)

