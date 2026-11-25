# Train model TEXT-ONLY (DistilBERT + mean pooling + head) cho A3
# Chạy từ thư mục gốc repo:
#   python -m scripts.a3_train_text configs/a3_text_only.yaml

import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from src.a3.data import NUM_CLASSES, ABODataset, TextCollator, class_names, load_split
from src.a3.models.text_only import TextClassifier

# ===== Đọc config =====
CONFIG = sys.argv[1] if len(sys.argv) > 1 else "configs/a3_text_only.yaml"
with open(CONFIG, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

SEED = cfg["seed"]
DATA_DIR = cfg["data"]["dir"]
MAX_LENGTH = cfg["data"]["max_length"]
MODEL_NAME = cfg["model"]["name"]
REVISION = cfg["model"]["revision"]
tr = cfg["train"]

OUT_DIR = f"{cfg['output_dir']}/text_only_seed{SEED}"
os.makedirs(OUT_DIR, exist_ok=True)
shutil.copy(CONFIG, f"{OUT_DIR}/config.yaml")   # lưu kèm config để biết kết quả này chạy bằng cấu hình nào
CKPT = f"{OUT_DIR}/best.pt"

torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Dữ liệu: dùng data.py chung của nhóm =====
splits = load_split(DATA_DIR)          # tự kiểm tra số dòng, nhãn, leakage
names = class_names(splits)
print({k: len(v) for k, v in splits.items()}, "| classes:", names)

tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION)
collate = TextCollator(tok, max_length=MAX_LENGTH)


def make_loader(split, shuffle=False, batch_size=128):
    # text-only: load_images=False để không mở file ảnh
    ds = ABODataset(splits[split], load_images=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def to_device(batch):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


train_loader = make_loader("train", shuffle=True, batch_size=tr["batch_size"])
val_loader = make_loader("validation")
test_loader = make_loader("test")

# ===== Model, optimizer =====
model = TextClassifier(MODEL_NAME, num_classes=NUM_CLASSES,
                       dropout=cfg["model"]["dropout"], revision=REVISION).to(device)
model.encoder.freeze()

# 2 nhóm lr: các layer cuối của encoder và head.
# Lúc encoder còn freeze thì không có gradient, AdamW tự bỏ qua nhóm đó.
last_layers = model.encoder.bert.transformer.layer[-tr["unfreeze_last"]:]
optimizer = torch.optim.AdamW([
    {"params": last_layers.parameters(), "lr": float(tr["lr_encoder"])},
    {"params": model.head.parameters(), "lr": float(tr["lr_head"])},
], weight_decay=tr["weight_decay"])
scheduler = get_cosine_schedule_with_warmup(optimizer, 0, tr["epochs"] * len(train_loader))
loss_fn = torch.nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(loader):
    model.eval()
    embs, logits_all, labels, ids = [], [], [], []
    for batch in loader:
        batch = to_device(batch)
        h = model.encoder(batch["input_ids"], batch["attention_mask"])   # [B, 768]
        logits = model.head(h)                                           # [B, 12]
        embs.append(h.cpu())
        logits_all.append(logits.cpu())
        labels.append(batch["labels"].cpu())
        ids += batch["item_id"]
    embs, logits_all, labels = torch.cat(embs), torch.cat(logits_all), torch.cat(labels)
    preds = logits_all.argmax(1)
    return {
        "loss": loss_fn(logits_all, labels).item(),
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
        "emb": embs, "logits": logits_all, "labels": labels, "preds": preds, "ids": ids,
    }


# ===== Train =====
best_f1, best_epoch, bad = -1, 0, 0
log = []
start = time.time()

for epoch in range(1, tr["epochs"] + 1):
    if epoch == tr["freeze_epochs"] + 1:
        model.encoder.unfreeze_last(tr["unfreeze_last"])
        print(f"-> Mở {tr['unfreeze_last']} layer cuối của encoder")

    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = to_device(batch)
        loss = loss_fn(model(batch), batch["labels"])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    val = evaluate(val_loader)
    train_loss = total_loss / len(train_loader)
    log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"],
                "val_acc": val["acc"], "val_f1": val["f1"]})
    print(f"epoch {epoch:2d} | train_loss {train_loss:.4f} | val_loss {val['loss']:.4f} "
          f"| val_acc {val['acc']:.4f} | val_f1 {val['f1']:.4f}")

    # lưu checkpoint theo val macro-F1, early stopping
    if val["f1"] > best_f1:
        best_f1, best_epoch, bad = val["f1"], epoch, 0
        torch.save(model.state_dict(), CKPT)
    else:
        bad += 1
        if bad >= tr["patience"]:
            print(f"Dừng sớm ở epoch {epoch}, tốt nhất là epoch {best_epoch}")
            break

train_time = time.time() - start
pd.DataFrame(log).to_csv(f"{OUT_DIR}/log.csv", index=False)

# ===== Test: chỉ chạy 1 lần với checkpoint tốt nhất =====
model.load_state_dict(torch.load(CKPT, map_location=device))
t = time.time()
test = evaluate(test_loader)
test_time = time.time() - t

print(classification_report(test["labels"], test["preds"], target_names=names, digits=4))
result = {
    "seed": SEED, "best_epoch": best_epoch, "best_val_f1": best_f1,
    "test_acc": test["acc"], "test_f1": test["f1"],
    "params": sum(p.numel() for p in model.parameters()),
    "train_time_s": round(train_time, 1), "test_time_s": round(test_time, 2),
}
print(result)
json.dump(result, open(f"{OUT_DIR}/result.json", "w"), indent=2)

# ===== Xuất embedding + logits cho Nam làm fusion =====
for name, loader in [("train", make_loader("train")),     # train không xáo trộn khi xuất
                     ("validation", val_loader), ("test", test_loader)]:
    out = test if name == "test" else evaluate(loader)
    np.save(f"{OUT_DIR}/text_emb_{name}.npy", out["emb"].numpy())
    np.save(f"{OUT_DIR}/text_logits_{name}.npy", out["logits"].numpy())
    pd.DataFrame({"item_id": out["ids"], "label": out["labels"].numpy()}).to_csv(
        f"{OUT_DIR}/ids_{name}.csv", index=False)
    print(name, "emb", tuple(out["emb"].shape), "logits", tuple(out["logits"].shape))