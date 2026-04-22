# Research Project: Large Language Models for Morphological Segmentation

## Description

This project investigates the applicability of large language models (LLMs) for the task of morphological segmentation. The goal is to explore how LLMs can learn and predict morpheme boundaries in words, providing insights into their performance on structured linguistic tasks.

## Setup

After cloning the repository, ensure all required libraries and dependencies are installed by running:

```bash
uv sync
```

This will automatically install the necessary packages for both training and inference.

## Test Run

To verify that the pipeline is functioning correctly, execute the command below:

```bash
uv run do-train --config-name smoke_train
```

After it you can test inference: change checkpoint path in `configs/inference/loader/local.yaml` confing, then execute this command:

```bash
uv run run-inference --config-name smoke_inference
```

If the scripts complete successfully, the pipeline is operational and ready for further experiments.

**Note**: execute all comands from dir root.