import hydra
import torch

from omegaconf import DictConfig
from transformers import AutoTokenizer


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def predict(cfg: DictConfig) -> None:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.cfg.model_name)

    model = hydra.utils.instantiate(cfg.model, tokinzer=tokenizer)

    checkpoint = torch.load(cfg.training.resume_from_checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    model.eval().cuda()

    with torch.no_grad():
        word = ""
        while True:
            prompt = ""
            inputs = tokenizer(prompt, return_tensor="pt").to(model.device)

            outputs = model.model.generate(
                **inputs, max_new_tokens=64, eos_token_id=tokenizer.eos_token_id
            )

            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"{word} -> {result.split('### Ответ:')[1].strip()}")


if __name__ == "__main__":
    predict()
