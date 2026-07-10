import argparse
import json
import os
import random
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer,
    GPTNeoForCausalLM,
    get_linear_schedule_with_warmup,
    logging as transformers_logging,
)

MODEL_NAME = "EleutherAI/gpt-neo-125M"
MAX_LENGTH = 256
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_STEPS = 0
PATIENCE = 2
SAMPLE_SIZE = 2000
LOG_DIR = "runs/gpt_neo_dolly"
SAVE_DIR = "saved_gpt_neo_dolly"
DEFAULT_DATASET = r"day 5\brick dataset\databricks-dolly-15k.jsonl"


def load_dolly_jsonl(path: str, sample_size: int = None):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            instruction = data.get("instruction", "").strip()
            context = data.get("context", "").strip()
            response = data.get("response", "").strip()
            if not instruction or not response:
                continue

            prompt = f"### Instruction:\n{instruction}\n"
            if context:
                prompt += f"\n### Context:\n{context}\n"
            prompt += "\n### Response:\n"
            examples.append((prompt, response))

    if sample_size is not None and sample_size < len(examples):
        examples = random.sample(examples, sample_size)
    return examples


class DollyDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = MAX_LENGTH
        self.examples = examples

        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        for prompt, response in self.examples:
            full_text = prompt + response
            encoding = tokenizer(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            prompt_encoding = tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            labels = encoding.input_ids.clone()
            prompt_len = prompt_encoding.input_ids.size(1)
            if prompt_len >= self.max_length:
                labels[:] = -100
            else:
                labels[:, :prompt_len] = -100

            self.input_ids.append(encoding.input_ids.squeeze(0))
            self.attention_masks.append(encoding.attention_mask.squeeze(0))
            self.labels.append(labels.squeeze(0))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }


def build_dataloaders(examples, tokenizer, batch_size: int):
    random.shuffle(examples)
    train_split = int(0.7 * len(examples))
    val_split = int(0.85 * len(examples))

    train_examples = examples[:train_split]
    valid_examples = examples[train_split:val_split]
    test_examples = examples[val_split:]

    train_dataset = DollyDataset(train_examples, tokenizer)
    valid_dataset = DollyDataset(valid_examples, tokenizer)
    test_dataset = DollyDataset(test_examples, tokenizer)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(valid_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size),
    )


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()

            logits = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            mask = labels != -100
            total_correct += (logits == labels).masked_select(mask).sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, accuracy


def train(args):
    transformers_logging.set_verbosity_info()
    print("Loading tokenizer (may download if needed)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_path = args.dataset_path or DEFAULT_DATASET
    print(f"Opening Dolly dataset at {dataset_path}...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    examples = load_dolly_jsonl(dataset_path, sample_size=args.sample_size)
    print(f"Loaded {len(examples)} Dolly examples from {dataset_path}")

    train_loader, valid_loader, test_loader = build_dataloaders(examples, tokenizer, batch_size=args.batch_size)
    print(f"Train batches: {len(train_loader)}, valid batches: {len(valid_loader)}, test batches: {len(test_loader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model (may download if needed)...")
    model = GPTNeoForCausalLM.from_pretrained(MODEL_NAME).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    writer = SummaryWriter(LOG_DIR)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        valid_loss, valid_acc = evaluate(model, valid_loader, device)
        epoch_time = time.perf_counter() - epoch_start

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Valid", valid_loss, epoch)
        writer.add_scalar("Accuracy/Valid", valid_acc, epoch)

        print(f"Epoch {epoch}/{args.epochs} - {epoch_time:.2f}s")
        print(f"  train loss: {train_loss:.4f}")
        print(f"  valid loss: {valid_loss:.4f}, valid acc: {valid_acc:.4f}")

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            patience_counter = 0
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"  Saved best model to {SAVE_DIR}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered")
                break

    writer.close()
    test_loss = evaluate(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f}")

    return model, tokenizer, device


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = GPTNeoForCausalLM.from_pretrained(args.model_dir).to(device)

    print(f"Loaded model from {args.model_dir} on {device}")
    print("Enter an instruction; press Enter on blank line to exit.")

    while True:
        text = input("Instruction: ").strip()
        if not text:
            break
        prompt = f"### Instruction:\n{text}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                top_p=0.95,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated[len(prompt) :].strip()
        print(f"\nResponse: {answer}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-Neo on Dolly 15k JSONL")
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET, help="Path to Dolly JSONL dataset")
    parser.add_argument("--sample_size", type=int, default=SAMPLE_SIZE, help="Number of examples to sample for training")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--infer", action="store_true", help="Run interactive inference using a saved model")
    parser.add_argument("--model_dir", default=SAVE_DIR, help="Saved model directory for inference")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.infer:
        infer(args)
    else:
        train(args)
