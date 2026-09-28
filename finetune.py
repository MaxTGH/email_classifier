"""Step 3: fine-tune DistilBERT (66M parameters) to classify messages.

Written as a plain PyTorch loop, not the Hugging Face Trainer, so every step
is visible: batch -> forward pass -> loss -> backward -> optimizer step.

  python 03_finetune.py              # full run
  python 03_finetune.py --quick      # small subset, just to check it works
  python 03_finetune.py --data path/to/emails.csv   # train on a CSV somewhere else
"""
import argparse
import time

import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 125    # tokens per message; longer ones are truncated
BATCH_SIZE = 8
EPOCHS = 3
LR = 5e-5        # small: we are nudging pretrained weights, not learning from scratch

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--data", default="data/emails.csv", help="CSV made by prepare_data.py")
args = ap.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# ---- Data -------------------------------------------------------------------
df = pd.read_csv(args.data)
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
if args.quick:
    train, test, EPOCHS = train.head(200), test.head(100), 1

labels = sorted(df["label"].unique())
label2id = {l: i for i, l in enumerate(labels)}

tok = AutoTokenizer.from_pretrained(MODEL_NAME)


def make_loader(frame, shuffle):
    pairs = list(zip(frame["text"], frame["label"].map(label2id)))

    def collate(batch):
        texts, ys = zip(*batch)
        enc = tok(list(texts), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        enc["labels"] = torch.tensor(ys)
        return enc

    return DataLoader(pairs, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=collate)


train_loader, test_loader = make_loader(train, True), make_loader(test, False)

# ---- Model ------------------------------------------------------------------
# Pretrained DistilBERT body + a brand-new, randomly initialised classification
# head with one output per label. Fine-tuning trains both together.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(labels), label2id=label2id, id2label=dict(enumerate(labels))
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)


def evaluate():
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds += logits.argmax(-1).tolist()
            gold += batch["labels"].tolist()
    return preds, gold


# ---- Train ------------------------------------------------------------------
print(f"Training on {len(train)} messages, testing on {len(test)}, device={device}\n")
for epoch in range(EPOCHS):
    model.train()
    start, running = time.time(), 0.0
    for step, batch in enumerate(train_loader, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss   # cross-entropy, computed because we passed labels
        loss.backward()              # gradients for all 66M parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        running += loss.item()
        if step % 20 == 0 or step == len(train_loader):
            print(f"  epoch {epoch + 1} step {step}/{len(train_loader)}  loss {running / step:.4f}")

    preds, gold = evaluate()
    acc = sum(p == g for p, g in zip(preds, gold)) / len(gold)
    print(f"Epoch {epoch + 1}: test accuracy {acc:.3f}  ({time.time() - start:.0f}s)\n")

print(classification_report(gold, preds, labels=range(len(labels)), target_names=labels, digits=3))

model.save_pretrained("model")
tok.save_pretrained("model")
print("Saved to ./model; try: python 04_predict.py \"your message here\"")
