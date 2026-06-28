import hydra

import torch.multiprocessing as mp

from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from datetime import datetime

from morphseg.utils import save_run_info


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def train(cfg: DictConfig) -> None:
    mp.set_start_method("spawn", force=True)

    print(OmegaConf.to_yaml(cfg))

    last_checkpoint_path = Path(cfg.run.dir) / "checkpoints" / "last.ckpt"
    experiment_run_dir = Path(cfg.run.dir).parent
    run_info = {
        "train": {
            "last_ckpt": str(last_checkpoint_path),
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "training_run_dir": str(Path.cwd().absolute()),
        }
    }

    save_run_info(run_info, str(experiment_run_dir))

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_cfg.model_name)

        model = hydra.utils.instantiate(
            cfg.model, tokenizer=tokenizer, _recursive_=False
        )
        datamodule = hydra.utils.instantiate(
            cfg.datamodule, tokenizer=tokenizer, _recursive_=False
        )

        trainer = hydra.utils.instantiate(
            cfg.trainer,
            logger=hydra.utils.instantiate(cfg.logger),
        )

        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.training.resume_from_checkpoint,
            weights_only=False,
        )

        run_info["train"]["status"] = "finished"
        run_info["train"]["best_ckpt"] = trainer.checkpoint_callback.best_model_path

    except Exception as e:
        run_info["train"]["status"] = "failed"
        run_info["train"]["error"] = str(e)
        print(e)
    finally:
        run_info["train"]["finished_at"] = datetime.now().isoformat()

        save_run_info(run_info, str(experiment_run_dir))

        print(f"{run_info=}")


if __name__ == "__main__":
    train()
