import hydra

import torch.multiprocessing as mp

from transformers import AutoTokenizer
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from datetime import datetime

from morphseg.utils import load_checkpoint_path, load_json, save_run_info


@hydra.main(version_base="1.3", config_path="../configs", config_name="test")
def test(cfg: DictConfig) -> None:
    mp.set_start_method("spawn", force=True)

    print(OmegaConf.to_yaml(cfg))

    try:
        best_ckpt_path = load_checkpoint_path(cfg.run_info_dir, "best_ckpt")
        run_info = load_json(str(Path(cfg.run_info_dir) / "run_info.json"))

        run_info["test"] = {
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "test_run_dir": str(Path.cwd().absolute()),
        }

        tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_cfg.model_name)

        model = hydra.utils.instantiate(
            cfg.model,
            tokenizer=tokenizer,
            ckpt_path=best_ckpt_path,
            _recursive_=False,
        )
        datamodule = hydra.utils.instantiate(
            cfg.datamodule, tokenizer=tokenizer, _recursive_=False
        )

        trainer = hydra.utils.instantiate(
            cfg.trainer,
            logger=hydra.utils.instantiate(cfg.logger),
        )

        results = trainer.test(
            model=model,
            datamodule=datamodule,
        )

        run_info["test"]["status"] = "finished"
        run_info["test"]["metrics"] = results
        print(results)
    except Exception as e:
        run_info["test"]["status"] = "failed"
        run_info["test"]["error"] = str(e)
        print(e)
    finally:
        run_info["test"]["finished_at"] = datetime.now().isoformat()
        save_run_info(run_info, cfg.run_info_dir)


if __name__ == "__main__":
    test()
