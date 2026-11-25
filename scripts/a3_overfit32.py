# Kiểm tra bug: model phải học thuộc được 32 mẫu (loss -> ~0, acc -> 100%)
# Chạy từ thư mục gốc repo: python -m scripts.a3_overfit32

import json
import pandas as pd
import torch
from transformers import AutoTokenizer
from src.a3.models.text_only import TextClassifier

MODEL_NAME = "distilbert/distilbert-base-uncased"
DATA_DIR = "data/processed/abo"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# 1. Lấy 32 mẫu train
df = pd.read_csv(f"{DATA_DIR}/train.csv").sample(32, random_state=42)
with open(f"{DATA_DIR}/labels.json") as f:
    num_classes = len(json.load(f))
print("Số lớp:", num_classes, "| nhãn trong 32 mẫu:", sorted(df["label"].unique()))

# 2. Tokenize thành 1 batch
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
batch = tok(list(df["title"]), padding=True, truncation=True, max_length=96, return_tensors="pt")
batch = {k: v.to(device) for k, v in batch.items()}
labels = torch.tensor(df["label"].values, dtype=torch.long, device=device)

# 3. Model: train hết, không freeze
model = TextClassifier(MODEL_NAME, num_classes=num_classes).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

# 4. Lặp 100 bước trên đúng 32 mẫu đó
model.train()
for step in range(1, 101):
    logits = model(batch)                  # [32, num_classes]
    loss = loss_fn(logits, labels)         # không softmax trước CrossEntropyLoss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        acc = (logits.argmax(dim=1) == labels).float().mean().item()
        print(f"step {step:3d} | loss {loss.item():.4f} | acc {acc:.2%}")