# Пайплайн обучения и инференса LLM

Данный репозиторий содержит инструменты для дообучения (Fine-tuning) языковых моделей с использованием методов PEFT (LoRA) и последующего инференса.

## Структура проекта

```text
.
├── configs/
│   ├── train.yaml    # Конфигурация параметров обучения
│   └── infer.yaml    # Конфигурация параметров инференса
├── data/
│   ├── train.jsonl   # Данные для обучения
│   └── test.jsonl    # Данные для тестирования
├── outputs/          # Директория для сохранения результатов
├── train.py          # Скрипт запуска обучения
├── inference.py      # Скрипт запуска инференса
├── ...
└── README.md
```

## Конфигурация

Управление параметрами осуществляется через YAML-файлы в директории `configs/`.

### 1. Конфигурация обучения (`configs/train.yaml`)

Файл определяет базовую модель, параметры LoRA, гиперпараметры обучения и пути к данным.

Пример конфигурации:

```yaml
model:
  name: microsoft/Phi-3-mini-4k-instruct  # Название модели из HuggingFace Hub (AutoModelForCausalLM)

lora:
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules: all-linear
  bias: none
  task_type: CAUSAL_LM

training:
  batch_size: 3
  grad_accum: 8
  lr: 2e-4
  epochs: 5
  bf16: true
  logging_steps: 50
  save_strategy: epoch

paths:
  train_data: data/train.jsonl
  test_data: data/test.jsonl
  output_dir: outputs

```

### 2. Конфигурация инференса (`configs/infer.yaml`)

Файл определяет параметры генерации и метрики. **Важно:** поле `run_id` необходимо для загрузки весов обученной модели.

Пример конфигурации:

```yaml
model:
  name: microsoft/Phi-3-mini-4k-instruct
  lora_path: final_model

paths:
  test_data: data/test.jsonl
  output_dir: outputs
  run_id: "20251231_004151"  # ID запуска обучения (соответствует папке в outputs)

generation:
  max_new_tokens: 64
  do_sample: false

metrics:
  - morpheme_precision_full
  - morpheme_recall_full
  - morpheme_f1_full
  - morpheme_precision_root
  - morpheme_recall_root
  - morpheme_f1_root
  - char_level_accuracy
  - word_accuracy

```

## Запуск обучения

Для запуска процесса обучения выполните команду:

```bash
python train.py
```

Скрипт автоматически загрузит конфигурацию из `configs/train.yaml`.

### Структура выходных данных

По завершении обучения (или в процессе, в зависимости от `save_strategy`) в директории `outputs` будет создана следующая иерархия:

```text
outputs/
    <normalized_model_name>/
        runs/
            <run_id>/
                train/
                    checkpoints/    # Промежуточные чекпоинты
                    final_model/    # Итоговые веса LoRA адаптера

```

* **<normalized_model_name>**: Имя модели, приведенное к формату пути файловой системы.
* **<run_id>**: Уникальный идентификатор запуска (timestamp), который необходимо использовать для инференса.

## Запуск инференса

Инференс осуществляется путем объединения базовой модели и обученного LoRA-адаптера.

1. Откройте файл `configs/infer.yaml`.
2. Убедитесь, что параметры `model.name` совпадают с использованными при обучении.
3. Установите значение `paths.run_id`, соответствующее конкретному запуску обучения (имя папки внутри `runs/`). Скрипт будет искать адаптер в подпапке `final_model`.
4. Запустите скрипт:

```bash
python inference.py
```

Скрипт загрузит базовую модель, применит веса из указанного `run_id` и выполнит генерацию на тестовых данных с расчетом указанных метрик.