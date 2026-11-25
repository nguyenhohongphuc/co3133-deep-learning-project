# Đo độ dài title theo token của DistilBERT để chọn max_length
# Chạy: python scripts/a3_token_lengths.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# ===== Cấu hình (sửa cho đúng máy mình) =====
SPLIT_DIR = "data/processed/abo"  # thư mục chứa train.csv, validation.csv, test.csv
MAX_LENGTH = 64
OUT_DIR = "results/a3"

os.makedirs(OUT_DIR + "/tables", exist_ok=True)
os.makedirs(OUT_DIR + "/figures", exist_ok=True)

tok = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# ===== Đọc dữ liệu và đếm token =====
data = {}
for name in ["train", "validation", "test"]:
    df = pd.read_csv(f"{SPLIT_DIR}/{name}.csv")
    df["title"] = df["title"].fillna("")

    # không truncation, để thấy được cả title dài hơn MAX_LENGTH
    df["ids"] = tok(list(df["title"]))["input_ids"]
    df["n_tokens"] = df["ids"].apply(len)            # đã gồm [CLS] và [SEP]
    df["n_words"] = df["title"].apply(lambda s: len(s.split()))
    df["has_unk"] = df["ids"].apply(lambda ids: tok.unk_token_id in ids)

    data[name] = df
    print(name, ":", len(df), "dòng")

# ===== Bảng thống kê =====
rows = []
for name, df in data.items():
    n = df["n_tokens"]
    rows.append({
        "split": name,
        "n": len(df),
        "min": n.min(),
        "median": n.median(),
        "p95": n.quantile(0.95),
        "p99": n.quantile(0.99),
        "max": n.max(),
        "token_per_word": round((n - 2).sum() / df["n_words"].sum(), 2),
        "%unk": round(df["has_unk"].mean() * 100, 2),
        "%>32": round((n > 32).mean() * 100, 2),
        "%>48": round((n > 48).mean() * 100, 2),
        "%>64": round((n > 64).mean() * 100, 2),
        "%>96": round((n > 96).mean() * 100, 2),
    })

table = pd.DataFrame(rows)
print(table)
table.to_csv(OUT_DIR + "/tables/a3_token_lengths.csv", index=False)

# ===== Xem phần bị mất của 5 title dài nhất (train) =====
train = data["train"]
print("\n--- 5 title dài nhất ---")
for _, row in train.sort_values("n_tokens", ascending=False).head(5).iterrows():
    print(row["n_tokens"], "tokens:", row["title"])
    if row["n_tokens"] > MAX_LENGTH:
        # giữ [CLS] + 62 token đầu + [SEP], phần còn lại bị bỏ
        print("   bị mất:", tok.decode(row["ids"][MAX_LENGTH - 1:-1]))

# ===== Title có [UNK] và từ lạ =====
print("\n--- Title có [UNK] ---")
for _, row in train[train["has_unk"]].head(5).iterrows():
    print(row["title"])
    print("  ", tok.convert_ids_to_tokens(row["ids"]))

print("\n--- Từ lạ ---")
for w in ["kattokyu", "howaitogo", "FL/209/BGE"]:
    print(w, "->", tok.tokenize(w))

# ===== Histogram =====
plt.hist(train["n_tokens"], bins=50)
plt.axvline(MAX_LENGTH, color="red", linestyle="--", label=f"max_length = {MAX_LENGTH}")
plt.xlabel("Số token / title")
plt.ylabel("Số title")
plt.title("Độ dài token của title (train)")
plt.legend()
plt.savefig(OUT_DIR + "/figures/a3_token_lengths.png", dpi=150)
print("\nĐã lưu kết quả vào", OUT_DIR)