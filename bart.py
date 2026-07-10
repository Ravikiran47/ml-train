
import os
import zipfile
import random
import numpy as np
import pandas as pd
import torch

from google.colab import files

from sklearn.model_selection import train_test_split

from transformers import BartTokenizer

from torch.utils.data import Dataset
from torch.utils.data import DataLoader



uploaded = files.upload()

zip_name = list(uploaded.keys())[0]

extract_dir = "/content/bbc_dataset"

with zipfile.ZipFile(zip_name, "r") as zip_ref:
    zip_ref.extractall(extract_dir)

print("Dataset Extracted Successfully!")


news_root = None
summary_root = None

for root, dirs, files in os.walk(extract_dir):

    if os.path.basename(root).lower() == "news articles":
        news_root = root

    if os.path.basename(root).lower() == "summaries":
        summary_root = root

print("News Folder :", news_root)
print("Summary Folder :", summary_root)



articles = []
summaries = []

categories = os.listdir(news_root)

print("Categories :", categories)

for category in categories:

    article_folder = os.path.join(news_root, category)
    summary_folder = os.path.join(summary_root, category)

    for filename in os.listdir(article_folder):

        article_path = os.path.join(article_folder, filename)
        summary_path = os.path.join(summary_folder, filename)

        if os.path.exists(summary_path):

            with open(article_path, encoding="latin1") as f:
                article = f.read()

            with open(summary_path, encoding="latin1") as f:
                summary = f.read()

            articles.append(article)
            summaries.append(summary)


df = pd.DataFrame({

    "article": articles,
    "summary": summaries

})

print(df.head())

print()

print("Total Samples :", len(df))



df = df.sample(

    n=1000,

    random_state=42

).reset_index(drop=True)

print("Samples Selected :", len(df))

train_df, temp_df = train_test_split(

    df,

    test_size=0.20,

    random_state=42

)

val_df, test_df = train_test_split(

    temp_df,

    test_size=0.50,

    random_state=42

)

print()

print("Training :", len(train_df))
print("Validation :", len(val_df))
print("Testing :", len(test_df))



MODEL_NAME = "facebook/bart-base"

MAX_INPUT_LENGTH = 512

MAX_TARGET_LENGTH = 128

BATCH_SIZE = 4

EPOCHS = 1

LEARNING_RATE = 5e-5

WARMUP_STEPS = 100



device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

print()

print("Device :", device)


tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer Loaded Successfully")



class BBCDataset(Dataset):

    def __init__(self, dataframe, tokenizer):

        self.data = dataframe.reset_index(drop=True)

        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        article = self.data.loc[idx, "article"]

        summary = self.data.loc[idx, "summary"]

        article_encoding = self.tokenizer(

            article,

            max_length=MAX_INPUT_LENGTH,

            padding="max_length",

            truncation=True,

            return_tensors="pt"

        )

        summary_encoding = self.tokenizer(

            summary,

            max_length=MAX_TARGET_LENGTH,

            padding="max_length",

            truncation=True,

            return_tensors="pt"

        )

        labels = summary_encoding["input_ids"].squeeze()

        labels[labels == tokenizer.pad_token_id] = -100

        return {

            "input_ids":

                article_encoding["input_ids"].squeeze(),

            "attention_mask":

                article_encoding["attention_mask"].squeeze(),

            "labels":

                labels

        }


train_dataset = BBCDataset(

    train_df,

    tokenizer

)

val_dataset = BBCDataset(

    val_df,

    tokenizer

)

test_dataset = BBCDataset(

    test_df,

    tokenizer

)

print()

print("Dataset Objects Created")


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True

)

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False

)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False

)

print()

print("Train Batches :", len(train_loader))
print("Validation Batches :", len(val_loader))
print("Test Batches :", len(test_loader))


sample_batch = next(iter(train_loader))

print()

print(sample_batch["input_ids"].shape)
print(sample_batch["attention_mask"].shape)
print(sample_batch["labels"].shape)

print()

from transformers import (
    BartForConditionalGeneration,
    get_linear_schedule_with_warmup
)

from torch.optim import AdamW

from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm


model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

model.to(device)

print("Model Loaded Successfully!")



optimizer = AdamW(

    model.parameters(),

    lr=LEARNING_RATE

)

print("Optimizer Created")

total_training_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(

    optimizer=optimizer,

    num_warmup_steps=WARMUP_STEPS,

    num_training_steps=total_training_steps

)

print("Scheduler Created")


writer = SummaryWriter("runs/bart_training")

print("TensorBoard Ready")



class EarlyStopping:

    def __init__(self, patience=2):

        self.patience = patience

        self.best_loss = float("inf")

        self.counter = 0

        self.early_stop = False

    def __call__(self, validation_loss):

        if validation_loss < self.best_loss:

            self.best_loss = validation_loss

            self.counter = 0

        else:

            self.counter += 1

            print(

                f"EarlyStopping Counter: {self.counter}/{self.patience}"

            )

            if self.counter >= self.patience:

                self.early_stop = True

early_stopping = EarlyStopping(patience=2)

train_losses = []

val_losses = []


for epoch in range(EPOCHS):

    print()

    print("=" * 60)

    print(f"Epoch {epoch+1}/{EPOCHS}")

    print("=" * 60)


    model.train()

    running_train_loss = 0

    train_progress = tqdm(train_loader)

    for batch in train_progress:

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(

            input_ids=input_ids,

            attention_mask=attention_mask,

            labels=labels

        )

        loss = outputs.loss

        running_train_loss += loss.item()

        loss.backward()

        optimizer.step()

        scheduler.step()

        train_progress.set_postfix(

            loss=loss.item()

        )

    avg_train_loss = running_train_loss / len(train_loader)

    train_losses.append(avg_train_loss)

    writer.add_scalar(

        "Loss/Train",

        avg_train_loss,

        epoch+1

    )

    print()

    print(f"Average Training Loss : {avg_train_loss:.4f}")

    model.eval()

    running_val_loss = 0

    with torch.no_grad():

        val_progress = tqdm(val_loader)

        for batch in val_progress:

            input_ids = batch["input_ids"].to(device)

            attention_mask = batch["attention_mask"].to(device)

            labels = batch["labels"].to(device)

            outputs = model(

                input_ids=input_ids,

                attention_mask=attention_mask,

                labels=labels

            )

            loss = outputs.loss

            running_val_loss += loss.item()

            val_progress.set_postfix(

                val_loss=loss.item()

            )

    avg_val_loss = running_val_loss / len(val_loader)

    val_losses.append(avg_val_loss)

    writer.add_scalar(

        "Loss/Validation",

        avg_val_loss,

        epoch+1

    )

    print()

    print(f"Average Validation Loss : {avg_val_loss:.4f}")

    early_stopping(avg_val_loss)

    if early_stopping.early_stop:

        print()

        print("Early Stopping Triggered")

        break

print()

print("="*60)

print("Training Completed")

print("="*60)

writer.close()

print()

print("Training Losses")

print(train_losses)

print()

print("Validation Losses")

print(val_losses)


import os
from transformers import BartTokenizer, BartForConditionalGeneration
SAVE_DIRECTORY = "/content/bart_bbc_model"

os.makedirs(SAVE_DIRECTORY, exist_ok=True)

model.save_pretrained(SAVE_DIRECTORY)
tokenizer.save_pretrained(SAVE_DIRECTORY)

print("="*60)
print("Model Saved Successfully!")
print("="*60)
loaded_tokenizer = BartTokenizer.from_pretrained(
    SAVE_DIRECTORY
)

loaded_model = BartForConditionalGeneration.from_pretrained(
    SAVE_DIRECTORY
)

loaded_model.to(device)

loaded_model.eval()

print("Saved Model Reloaded Successfully!")
sample_article = test_df.iloc[0]["article"]

actual_summary = test_df.iloc[0]["summary"]

print("="*60)
print("ARTICLE")
print("="*60)

print(sample_article[:1000])

print()

print("="*60)
print("ACTUAL SUMMARY")
print("="*60)

print(actual_summary)

inputs = loaded_tokenizer(

    sample_article,

    max_length=MAX_INPUT_LENGTH,

    truncation=True,

    return_tensors="pt"

)

input_ids = inputs["input_ids"].to(device)

attention_mask = inputs["attention_mask"].to(device)
summary_ids = loaded_model.generate(

    input_ids=input_ids,

    attention_mask=attention_mask,

    max_length=MAX_TARGET_LENGTH,

    min_length=30,

    num_beams=4,

    early_stopping=True,

    no_repeat_ngram_size=2

)

generated_summary = loaded_tokenizer.decode(

    summary_ids[0],

    skip_special_tokens=True

)

print()

print("="*60)
print("GENERATED SUMMARY")
print("="*60)

print(generated_summary)
print()

print("="*80)
print("COMPARISON")
print("="*80)

print()

print("Actual Summary\n")
print(actual_summary)

print()

print("-"*80)

print()

print("Generated Summary\n")
print(generated_summary)
print()

print("="*60)
print("LOSS VALUES")
print("="*60)

print("Training Loss :", train_losses)

print("Validation Loss :", val_losses)
%load_ext tensorboard

print()

print("Run the next cell to open TensorBoard.")