from transformers import DataCollatorForSeq2Seq


class DataCollatorForValidation(DataCollatorForSeq2Seq):
    def __call__(self, features):
        batch_targets = [f.pop("target") for f in features if "target" in f]
        batch_words = [f.pop("word") for f in features if "word" in f]

        batch = super().__call__(features)

        batch["target"] = batch_targets
        batch["word"] = batch_words

        return batch
