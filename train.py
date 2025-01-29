import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # for ultra-fast downloads

from dataclasses import asdict, dataclass

import evaluate
import numpy as np
import simple_parsing
import torch
import wandb
from datasets import load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn


@dataclass
class Config:
    debug: bool = False
    seed: int = 42
    out_dir: str = "./artifacts"
    wandb_project: str = "hf-wandb"
    wandb_enable: bool = True
    wandb_run_name: str = "qlora"
    model_id: str = "Qwen/Qwen2.5-3B"
    model_dtype: str = "bfloat16"
    attention_implementation: str = "flash_attention_2"
    quantize_to_4bit: bool = True
    max_seq_length: int = 512
    batch_size: int = 8
    num_epochs: int = 3
    learning_rate: float = 5e-5
    gradient_accumulation_steps: int = 1
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    save_artifacts: bool = True


def print_line():
    print("\n" + "#" + "-" * 100 + "#" + "\n")


def main(args: Config):
    set_seed(args.seed)
    if args.wandb_enable:
        wandb.init(
            name=args.wandb_run_name,
            project=args.wandb_project,
            config=asdict(args),
        )

    os.makedirs(args.out_dir, exist_ok=True)

    # load the dataset
    ds = load_dataset("imdb")
    label_list = ds["train"].features["label"].names
    num_labels = len(label_list)
    train_ds, eval_ds = ds["train"], ds["test"]

    if args.debug:
        train_ds = train_ds.shuffle(seed=42).select(range(50))
        eval_ds = train_ds.shuffle(seed=42).select(range(50))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="left")

    print("Loading and quantizing model...")
    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float16,
        attn_implementation=args.attention_implementation,
    )

    if args.quantize_to_4bit:
        print("Quantizing model to 4-bit")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=args.model_dtype,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, use_cache=False, num_labels=num_labels, **model_kwargs
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    def tokenize_func(example):
        return tokenizer(
            example["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=args.max_seq_length,
        )

    print("Preparing model for PEFT")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        modules_to_save=["score"],
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    print(model)

    train_ds = train_ds.map(tokenize_func, batched=True, desc="Tokenizing train data")
    eval_ds = eval_ds.map(tokenize_func, batched=True, desc="Tokenizing eval data")

    print_line()
    rand_idx = np.random.randint(0, len(train_ds))
    print(train_ds[rand_idx])
    print("\nFormatted Text:\n")
    print(tokenizer.decode(train_ds[rand_idx]["input_ids"]))
    print_line()

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=16)

    # Load all metrics
    accuracy = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # Compute all metrics
        accuracy_score = accuracy.compute(predictions=predictions, references=labels)[
            "accuracy"
        ]
        f1 = f1_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )["f1"]
        precision = precision_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )["precision"]
        recall = recall_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )["recall"]

        return {
            "accuracy": accuracy_score,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=1,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        bf16_full_eval=torch.cuda.is_bf16_supported(),
        fp16_full_eval=not torch.cuda.is_bf16_supported(),
        report_to="wandb" if args.wandb_enable else None,
        gradient_checkpointing=True,
        group_by_length=True,
        torch_compile=True,
        max_grad_norm=1.0,
        weight_decay=0.01,
    )
    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(args.out_dir)

    if args.wandb_enable:
        if args.save_artifacts:
            model_artifact = wandb.Artifact(name=args.wandb_run_name, type="model")
            model_artifact.add_dir(args.out_dir)
            wandb.log_artifact(model_artifact)
        wandb.finish()


if __name__ == "__main__":
    args = simple_parsing.parse(Config)
    main(args)
