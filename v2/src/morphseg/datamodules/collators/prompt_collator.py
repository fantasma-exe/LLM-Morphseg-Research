from transformers import DataCollatorForSeq2Seq


class DataCollatorContainingPrompt(DataCollatorForSeq2Seq):
    """
    Data collator that preserves raw prompt text in the output batch.

    This collator extends ``DataCollatorForSeq2Seq`` by extracting the
    ``"prompt_raw_text"`` field from each feature before collation and
    reinserting it into the resulting batch. This is necessary because
    standard Hugging Face data collators only operate on tensor-based
    fields such as ``input_ids``, ``labels``, and ``attention_mask``,
    and would otherwise discard auxiliary non-tensor data.

    Parameters
    ----------
    features : list[dict]
        A list of feature dictionaries. Each dictionary is expected to
        contain a key ``"prompt_raw_text"`` along with standard model
        inputs (e.g., ``input_ids``, ``labels``, ``attention_mask``).

    Returns
    -------
    dict
        A batch dictionary produced by ``DataCollatorForSeq2Seq`` with
        an additional key:
        
        - ``"prompt_raw_text"`` : list of str
            The raw prompt texts corresponding to each example in the batch,
            preserved in the original order.

    Raises
    ------
    RuntimeError
        If any feature in the input list does not contain the
        ``"prompt_raw_text"`` key.

    Notes
    -----
    This collator assumes that all input features contain
    ``"prompt_raw_text"``. Missing values will result in an error to
    prevent silent data inconsistency.
    """
    
    def __call__(self, features):
        batch_prompts = [f.pop("prompt_raw_text", None) for f in features]
        batch_prompts_clear = [p for p in batch_prompts if p is not None]
        
        if len(batch_prompts_clear) != len(batch_prompts):
            raise RuntimeError("Invalid batch in data collator")
        
        batch = super().__call__(features)
        batch["prompt_raw_text"] = batch_prompts
        
        return batch