from dataclasses import dataclass, field
from datasets import load_dataset
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertForSequenceClassification, AutoModelForSequenceClassification
from transformers import HfArgumentParser
import evaluate
from transformers import EvalPrediction
from transformers import Trainer, TrainingArguments
import wandb
from evaluate import load


@dataclass
class ScriptArguments:
    num_train_epochs: int = 3
    lr: float = 2e-5
    batch_size: int = 16
    max_train_samples: int = -1
    max_eval_samples: int = -1
    max_predict_samples: int = -1
    do_train: bool = False
    do_predict: bool = False
    model_path: str = "bert-base-uncased"


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    metric = load("accuracy")
    dataset = load_dataset("nyu-mll/glue", "mrpc")
    config = AutoConfig.from_pretrained(args.model_path)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return {"accuracy": metric.compute(predictions=preds, references=p.label_ids)["accuracy"]}

    def tokenize_function(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    device = torch.device("cuda")
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # global tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, config=config)
    model.to(device)
    # model = BertForSequenceClassification.from_pretrained(args.model_path, num_labels=2)

    dataset_tokenized = dataset.map(tokenize_function, batched=True, batch_size=len(dataset))
    train_dataset = dataset_tokenized['train']
    validation_dataset = dataset_tokenized['validation']
    test_dataset = dataset_tokenized['test']

    metric = load("accuracy")

    training_args = TrainingArguments(
        output_dir="./anlp_ex1_results/results",
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        report_to="wandb",
        run_name="ex1_version4",
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    if args.do_train:
        trainer.train()
        # model.save_pretrained("./anlp_ex1_results/models/")
        # trainer.save_model("./anlp_ex1_results/models/version1_model_trainer")
        # tokenizer.save_pretrained("./anlp_ex1_results/models/")



    if args.do_predict:
        model.eval()
        model.to(training_args.device)
        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        with open("predictions.txt", "w") as f:
            for pred in preds:
                f.write(str(pred) + "\n")




if __name__ == "__main__":
    main()
