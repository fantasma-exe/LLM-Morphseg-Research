import typing as tp

from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import DictConfig

from morphseg.inference.input import BaseInput
from morphseg.inference.output import BaseOutput
from morphseg.inference.predictor import Predictor


class InferencePipeline:
    """
    Pipeline for running inference on a set of input words using a predictor
    and saving or displaying the results via a specified output strategy.

    This class integrates input reading, prompt generation, batched prediction,
    and output handling.

    Parameters
    ----------
    dataset_factory: tp.Callable
        ...

    input_strategy : BaseInput
        Strategy for reading input data (e.g., from file or console).

    output_strategy : BaseOutput
        Strategy for saving or displaying output results (e.g., file or console).

    predictor : Predictor
        Predictor object used to generate model outputs from prompts.

    prompt_template : str
        Template string for generating prompts from input words.

    dataloader_kwargs : DictConfig
        Additional keyword arguments for the PyTorch DataLoader.
    """

    def __init__(
        self,
        dataset_factory: tp.Callable,
        input_strategy: BaseInput,
        output_strategy: BaseOutput,
        predictor: Predictor,
        prompt_template: str,
        dataloader_kwargs: DictConfig,
    ) -> None:
        self.dataset_factory = dataset_factory
        self.input_strategy = input_strategy
        self.output_strategy = output_strategy
        self.predictor = predictor
        self.prompt_template = prompt_template
        self.dataloader_kwargs = dataloader_kwargs

    def run(self) -> None:
        """
        Execute the inference pipeline.

        Steps
        -----
        1. Read input words using the input strategy.
        2. Generate prompts using the prompt template.
        3. Create a DataLoader for batched processing.
        4. Predict outputs in batches using the predictor.
        5. Save or display results using the output strategy.

        Notes
        -----
        If no input words are read, the pipeline prints a message and returns
        without performing inference.
        """
        words = self.input_strategy.read()
        if not words:
            print("-- No data for inference --")
            return

        def collate_fn(batch):
            batch_words = [item[0] for item in batch]
            batch_prompts = [item[1] for item in batch]
            return batch_words, batch_prompts

        inference_dataset = self.dataset_factory(
            words=words, prompt_template=self.prompt_template
        )

        dataloader = DataLoader(
            inference_dataset,
            collate_fn=collate_fn,
            **self.dataloader_kwargs,  # type: ignore
        )

        all_words = []
        all_preds = []

        for batch_words, batch_prompts in tqdm(dataloader, desc="Predicting"):
            preds = self.predictor.predict_batch(batch_prompts)
            all_words.extend(batch_words)
            all_preds.extend(preds)

        self.output_strategy.write(all_words, all_preds)
