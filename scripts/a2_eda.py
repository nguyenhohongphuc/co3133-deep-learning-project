#!/usr/bin/env python3
"""
A2 Dataset Proposal EDA - 30VNFoods (works for any ImageFolder-style dataset).

Covers the "must run yourself" items of the proposal (handbook S18, S19.1, S22):
  3. sample counts per split / per class
  4. preliminary distribution analysis (class balance, image sizes, formats)
  6. leakage check: exact + near-duplicate images within / across splits, cross-label duplicates
  1. dataset fingerprint (checksum) for the "version" field
  5/11. proposed removal list + resulting split sizes

Expected layout (auto-detected):
    <root>/<split>/<class>/<image>     e.g. Train/, Validate/, Test/
    <root>/<class>/<image>             (no predefined split -> single split "all")

Outputs (default: results/a2/eda/):
    summary.md                tables ready to paste into the proposal
    summary.json              all key numbers (machine-readable)
    counts_per_class.csv      images per class x split
    image_stats.csv           one row per image: split, class, size, format, mode, md5, phash
    corrupted.csv             files that failed to open / decode
    near_duplicate_pairs.csv  pairs with pHash Hamming distance <= max(--sens-thresholds)
    duplicate_clusters.csv    clusters at --threshold (single-linkage over pairs)
    leakage_drop_list.csv     proposed removals with reason
    figures/class_distribution.png, image_sizes.png, samples_per_class.png, duplicate_examples.png

Usage:
    pip install pillow imagehash numpy pandas matplotlib tqdm
    python scripts/a2_eda.py --root data/30VNFoods
    python scripts/a2_eda.py --root data/30VNFoods --threshold 4 --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import random
import sys
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    import imagehash
except ImportError:  # pragma: no cover
    sys.exit("Missing dependency: pip install imagehash")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **_):
        return x

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

IMG_EXT = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
SEED = 42


# --------------------------------------------------------------------------- #
# 1. Discover splits and files
# --------------------------------------------------------------------------- #
def _has_images(d: Path) -> bool:
    return any(f.is_file() and f.suffix.lower() in IMG_EXT for f in d.iterdir())


def _is_split_dir(d: Path) -> bool:
    """A split dir contains class sub-dirs that contain images."""
    return any(c.is_dir() and _has_images(c) for c in d.iterdir())


def find_splits(root: Path, splits_arg: list[str] | None) -> dict[str, Path]:
    if splits_arg:
        out = {s: root / s for s in splits_arg}
        missing = [str(p) for p in out.values() if not p.is_dir()]
        if missing:
            sys.exit(f"Split dir(s) not found: {missing}")
        return out
    subdirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    split_dirs = {d.name: d for d in subdirs if _is_split_dir(d)}
    if split_dirs:
        return split_dirs
    if subdirs and any(_has_images(d) for d in subdirs):
        return {"all": root}  # root/<class>/<img>
    if len(subdirs) == 1:  # e.g. Kaggle extracts into an extra wrapper folder
        return find_splits(subdirs[0], None)
    sys.exit(f"Could not detect dataset layout under {root}. Use --splits.")


def split_role(name: str) -> str:
    n = name.lower()
    if "test" in n:
        return "test"
    if "val" in n or "dev" in n:
        return "val"
    if "train" in n:
        return "train"
    return "other"


# Higher = more protected. Copies are removed from the lower-priority split.
ROLE_PRIORITY = {"test": 3, "val": 2, "other": 1, "train": 0}


def list_images(split_dirs: dict[str, Path]):
    rows, other_files = [], 0
    for split, sdir in split_dirs.items():
        for f in sdir.iterdir():
            if f.is_file():
                other_files += 1  # stray files directly in the split folder
        for cdir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            for f in sorted(cdir.rglob("*")):
                if not f.is_file():
                    continue
                if f.suffix.lower() in IMG_EXT:
                    rows.append((split, cdir.name, str(f)))
                else:
                    other_files += 1
    return rows, other_files


# --------------------------------------------------------------------------- #
# 2. Per-image analysis (runs in worker processes)
# --------------------------------------------------------------------------- #
def analyze_one(path: str) -> dict:
    rec = {"path": path, "bytes": None, "md5": None, "width": None, "height": None,
           "format": None, "mode": None, "phash": None, "ok": False, "error": ""}
    try:
        data = Path(path).read_bytes()
        rec["bytes"] = len(data)
        rec["md5"] = hashlib.md5(data).hexdigest()
        with Image.open(io.BytesIO(data)) as im:
            rec["width"], rec["height"] = im.size  # true size, before draft()
            rec["format"], rec["mode"] = im.format, im.mode
            if im.format == "JPEG":
                im.draft("RGB", (128, 128))  # fast reduced decode; pHash works at 32x32 anyway
            rgb = im.convert("RGB")  # forces full decode -> catches truncated files
        rec["phash"] = str(imagehash.phash(rgb, hash_size=8))
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


# --------------------------------------------------------------------------- #
# 3. Near-duplicate search (vectorised Hamming distance on 64-bit pHash)
# --------------------------------------------------------------------------- #
_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):  # numpy >= 2.0
        return np.bitwise_count(x)
    x = np.ascontiguousarray(x)
    return _POP8[x.view(np.uint8)].reshape(*x.shape, 8).sum(-1)


def near_pairs(h: np.ndarray, max_d: int, chunk: int = 256):
    n = len(h)
    I, J, D = [], [], []
    for s in tqdm(range(0, n, chunk), desc="pHash pairs", unit="chunk"):
        a, b = h[s:s + chunk], h[s:]
        d = popcount64(a[:, None] ^ b[None, :])
        ii, jj = np.nonzero(d <= max_d)
        keep = jj > ii  # j global = jj + s, i global = ii + s
        ii, jj = ii[keep], jj[keep]
        I.append(ii + s)
        J.append(jj + s)
        D.append(d[ii, jj])
    if not I:
        return np.array([], int), np.array([], int), np.array([], int)
    return np.concatenate(I), np.concatenate(J), np.concatenate(D).astype(int)


def cluster_ids(n: int, I, J) -> np.ndarray:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(I.tolist(), J.tolist()):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)])


# --------------------------------------------------------------------------- #
# 4. Figures
# --------------------------------------------------------------------------- #
def load_thumb(path: str, size: int = 160):
    with Image.open(path) as im:
        if im.format == "JPEG":
            im.draft("RGB", (size * 2, size * 2))
        im = im.convert("RGB")
        im.thumbnail((size, size))
        return np.asarray(im)


def fig_class_distribution(counts: pd.DataFrame, splits: list[str], out: Path):
    order = counts["total"].sort_values().index
    c = counts.loc[order, splits]
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(c) + 1.5))
    left = np.zeros(len(c))
    for s in splits:
        ax.barh(c.index, c[s], left=left, label=s)
        left += c[s].to_numpy()
    for y, v in enumerate(left):
        ax.text(v, y, f" {int(v)}", va="center", fontsize=7)
    ax.set_xlabel("Number of images")
    ax.set_title("Images per class (stacked by split)")
    ax.legend(loc="lower right")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_image_sizes(ok: pd.DataFrame, out: Path):
    w, h = ok["width"].to_numpy(float), ok["height"].to_numpy(float)
    short = np.minimum(w, h)
    aspect = w / h
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].scatter(w, h, s=3, alpha=0.3)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("width (px)"); axes[0].set_ylabel("height (px)")
    axes[0].set_title("Image width vs height (log-log)")
    axes[1].hist(np.clip(aspect, 0.25, 4), bins=np.logspace(np.log10(0.25), np.log10(4), 50))
    axes[1].set_xscale("log"); axes[1].set_xlabel("aspect ratio w/h (clipped 0.25-4)")
    axes[1].set_title("Aspect ratio")
    bins = np.logspace(np.log10(max(short.min(), 1)), np.log10(short.max() + 1), 50)
    axes[2].hist(short, bins=bins)
    axes[2].axvline(224, color="red", ls="--", label="224 px (model input)")
    axes[2].set_xscale("log"); axes[2].set_xlabel("shorter side (px)")
    axes[2].set_title("Shorter side"); axes[2].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_samples(ok: pd.DataFrame, split: str, out: Path, k: int = 4, blocks: int = 3):
    rng = random.Random(SEED)
    sub = ok[ok["split"] == split]
    classes = sorted(sub["class"].unique())
    rows = int(np.ceil(len(classes) / blocks))
    fig, axes = plt.subplots(rows, blocks * k, figsize=(blocks * k * 1.45, rows * 1.65))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for ci, cls in enumerate(classes):
        paths = sub.loc[sub["class"] == cls, "path"].tolist()
        pick = rng.sample(paths, min(k, len(paths)))
        r, b = ci // blocks, ci % blocks
        for j, p in enumerate(pick):
            ax = axes[r, b * k + j]
            try:
                ax.imshow(load_thumb(p))
            except Exception:  # noqa: BLE001
                pass
            if j == 0:
                ax.set_title(cls, fontsize=8, loc="left")
    fig.suptitle(f"Random samples per class ({split}, seed={SEED})")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_duplicates(pairs: pd.DataFrame, out: Path, max_pairs: int = 12):
    if pairs.empty:
        return False
    pairs = pairs.sort_values(["cross_label", "cross_split", "distance"],
                              ascending=[False, False, True]).head(max_pairs)
    rows = int(np.ceil(len(pairs) / 2))
    fig, axes = plt.subplots(rows, 4, figsize=(12, rows * 3.1))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for n, (_, r) in enumerate(pairs.iterrows()):
        row, col = n // 2, (n % 2) * 2
        for k, side in enumerate(("a", "b")):
            ax = axes[row, col + k]
            try:
                ax.imshow(load_thumb(r[f"path_{side}"], 220))
            except Exception:  # noqa: BLE001
                pass
            ax.set_title(f"{r[f'split_{side}']} / {r[f'class_{side}']}"
                         + (f"\nd={r['distance']}" if k == 0 else "\n"), fontsize=8)
    fig.suptitle("Near-duplicate pairs to verify manually (cross-label and cross-split first)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #
def md_table(df: pd.DataFrame, index=True) -> str:
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.tolist()) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="dataset root folder")
    ap.add_argument("--out", type=Path, default=Path("results/a2/eda"))
    ap.add_argument("--splits", nargs="*", help="split folder names (default: auto-detect)")
    ap.add_argument("--threshold", type=int, default=4,
                    help="pHash Hamming distance (0-64) treated as near-duplicate (default 4)")
    ap.add_argument("--sens-thresholds", type=int, nargs="*", default=[0, 2, 4, 6, 8, 10],
                    help="thresholds reported in the sensitivity table")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--max-images", type=int, default=0, help="debug: only scan the first N images")
    args = ap.parse_args()

    t_start = time.time()
    root = args.root.resolve()
    out = args.out
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    split_dirs = find_splits(root, args.splits)
    role_order = {"train": 0, "val": 1, "other": 2, "test": 3}
    split_dirs = dict(sorted(split_dirs.items(), key=lambda kv: (role_order[split_role(kv[0])], kv[0])))
    splits = list(split_dirs)
    print(f"[1/6] Splits detected: { {s: str(p) for s, p in split_dirs.items()} }")

    files, other_files = list_images(split_dirs)
    if args.max_images:
        files = files[: args.max_images]
    if not files:
        sys.exit("No images found.")
    print(f"      {len(files)} image files, {other_files} non-image files ignored")

    # ---- per-image analysis
    print(f"[2/6] Reading images with {args.workers} workers ...")
    paths = [f[2] for f in files]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        recs = list(tqdm(ex.map(analyze_one, paths, chunksize=64), total=len(paths), unit="img"))
    df = pd.DataFrame(recs)
    df.insert(0, "split", [f[0] for f in files])
    df.insert(1, "class", [f[1] for f in files])
    df["relpath"] = [os.path.relpath(p, root) for p in df["path"]]
    df.drop(columns=["path"]).to_csv(out / "image_stats.csv", index=False)

    bad = df[~df["ok"]]
    bad[["split", "class", "relpath", "bytes", "error"]].to_csv(out / "corrupted.csv", index=False)
    ok = df[df["ok"]].reset_index(drop=True)
    print(f"      readable: {len(ok)}, corrupted/unreadable: {len(bad)}")

    # Dataset fingerprint: sha256 over sorted "relpath:md5" (use it as the version checksum)
    fp_src = "\n".join(sorted(f"{r}:{m}" for r, m in zip(df["relpath"], df["md5"].fillna("NA"))))
    fingerprint = hashlib.sha256(fp_src.encode()).hexdigest()

    # ---- counts
    print("[3/6] Counting ...")
    counts = (ok.groupby(["class", "split"]).size().unstack(fill_value=0)
              .reindex(columns=splits, fill_value=0))
    counts["total"] = counts.sum(axis=1)
    counts = counts.sort_values("total", ascending=False)
    counts.to_csv(out / "counts_per_class.csv")

    split_rows = []
    for s in splits:
        c = counts[s][counts[s] > 0]
        split_rows.append({
            "split": s, "role": split_role(s), "images": int(c.sum()), "classes": int(len(c)),
            "min/class": int(c.min()) if len(c) else 0,
            "min class": c.idxmin() if len(c) else "",
            "max/class": int(c.max()) if len(c) else 0,
            "max class": c.idxmax() if len(c) else "",
            "mean/class": round(float(c.mean()), 1) if len(c) else 0,
            "imbalance (max/min)": round(float(c.max() / c.min()), 2) if len(c) else 0,
        })
    split_tbl = pd.DataFrame(split_rows).set_index("split")
    total_images = int(len(ok))

    # ---- size / format stats
    w, h = ok["width"].astype(float), ok["height"].astype(float)
    short = np.minimum(w, h)
    size_stats = {
        "width_min": int(w.min()), "width_median": int(w.median()), "width_max": int(w.max()),
        "height_min": int(h.min()), "height_median": int(h.median()), "height_max": int(h.max()),
        "short_side_median": int(np.median(short)),
        "pct_short_side_lt_224": round(float((short < 224).mean() * 100), 2),
        "pct_short_side_lt_128": round(float((short < 128).mean() * 100), 2),
        "aspect_ratio_median": round(float(np.median(w / h)), 3),
        "pct_aspect_outside_0.5_2": round(float((((w / h) < 0.5) | ((w / h) > 2)).mean() * 100), 2),
        "total_size_MB": round(float(df["bytes"].fillna(0).sum() / 2**20), 1),
        "file_size_median_KB": round(float(ok["bytes"].median() / 1024), 1),
    }
    formats = dict(Counter(ok["format"]))
    modes = dict(Counter(ok["mode"]))

    # ---- duplicates
    print("[4/6] Searching exact and near duplicates ...")
    exact_groups = ok.groupby("md5").filter(lambda g: len(g) > 1)
    n_exact_groups = int(exact_groups["md5"].nunique())

    hashes = np.array([int(x, 16) for x in ok["phash"]], dtype=np.uint64)
    max_d = max([args.threshold] + list(args.sens_thresholds))
    I, J, D = near_pairs(hashes, max_d)
    sp, cl = ok["split"].to_numpy(), ok["class"].to_numpy()
    pairs = pd.DataFrame({
        "idx_a": I, "idx_b": J, "distance": D,
        "split_a": sp[I], "class_a": cl[I], "split_b": sp[J], "class_b": cl[J],
        "path_a": ok["path"].to_numpy()[I], "path_b": ok["path"].to_numpy()[J],
    })
    pairs["cross_split"] = pairs["split_a"] != pairs["split_b"]
    pairs["cross_label"] = pairs["class_a"] != pairs["class_b"]
    pairs_out = pairs.assign(relpath_a=[os.path.relpath(p, root) for p in pairs["path_a"]],
                             relpath_b=[os.path.relpath(p, root) for p in pairs["path_b"]])
    pairs_out.drop(columns=["path_a", "path_b", "idx_a", "idx_b"]).sort_values("distance") \
        .to_csv(out / "near_duplicate_pairs.csv", index=False)

    sens = []
    for t in sorted(set(args.sens_thresholds) | {args.threshold}):
        p = pairs[pairs["distance"] <= t]
        sens.append({"threshold": t, "pairs": len(p),
                     "same split+label": int((~p.cross_split & ~p.cross_label).sum()),
                     "cross-split": int(p.cross_split.sum()),
                     "cross-label": int(p.cross_label.sum())})
    sens_tbl = pd.DataFrame(sens).set_index("threshold")

    main_pairs = pairs[pairs["distance"] <= args.threshold]
    cid = cluster_ids(len(ok), main_pairs["idx_a"].to_numpy(), main_pairs["idx_b"].to_numpy())
    ok["cluster"] = cid
    sizes = ok.groupby("cluster").size()
    multi = sizes[sizes > 1].index
    clus = ok[ok["cluster"].isin(multi)].copy()

    # ---- proposed removals
    print("[5/6] Building leakage drop list ...")
    drops = []
    for _, r in bad.iterrows():
        drops.append({"split": r["split"], "class": r["class"], "relpath": r["relpath"],
                      "cluster": -1, "reason": "corrupted_or_unreadable", "action": "drop"})
    clus_summary = []
    for c, g in clus.groupby("cluster"):
        labels, spl = sorted(g["class"].unique()), sorted(g["split"].unique())
        clus_summary.append({"cluster": int(c), "size": len(g), "splits": "|".join(spl),
                             "classes": "|".join(labels), "cross_split": len(spl) > 1,
                             "cross_label": len(labels) > 1})
        if len(labels) > 1:  # same picture, different label -> label noise, remove all copies
            for _, r in g.iterrows():
                drops.append({"split": r["split"], "class": r["class"], "relpath": r["relpath"],
                              "cluster": int(c), "reason": "cross_label_duplicate", "action": "drop"})
            continue
        if len(spl) > 1:  # leakage: keep the copy in the most protected split (test > val > train)
            keep_split = max(spl, key=lambda s: ROLE_PRIORITY[split_role(s)])
            for _, r in g[g["split"] != keep_split].iterrows():
                drops.append({"split": r["split"], "class": r["class"], "relpath": r["relpath"],
                              "cluster": int(c), "reason": f"leak_copy_of_{keep_split}", "action": "drop"})
            g = g[g["split"] == keep_split]
        if len(g) > 1:  # redundant copies inside one split: optional dedup, keep the first
            for _, r in g.iloc[1:].iterrows():
                drops.append({"split": r["split"], "class": r["class"], "relpath": r["relpath"],
                              "cluster": int(c), "reason": "within_split_duplicate", "action": "optional"})
    clus_df = pd.DataFrame(clus_summary, columns=["cluster", "size", "splits", "classes",
                                                  "cross_split", "cross_label"])
    clus_df.to_csv(out / "duplicate_clusters.csv", index=False)
    drop_df = pd.DataFrame(drops, columns=["split", "class", "relpath", "cluster", "reason", "action"])
    drop_df.to_csv(out / "leakage_drop_list.csv", index=False)

    required = drop_df[drop_df["action"] == "drop"]
    after = []
    for s in splits:
        n0 = int((df["split"] == s).sum())
        nd = int((required["split"] == s).sum())
        no = int(((drop_df["split"] == s) & (drop_df["action"] == "optional")).sum())
        after.append({"split": s, "files": n0, "drop (required)": nd,
                      "after drop": n0 - nd, "optional dedup": no})
    after_tbl = pd.DataFrame(after).set_index("split")
    reason_counts = drop_df.groupby(["reason", "action"]).size().rename("images").reset_index()

    # ---- figures
    print("[6/6] Drawing figures ...")
    fig_class_distribution(counts, splits, figdir / "class_distribution.png")
    fig_image_sizes(ok, figdir / "image_sizes.png")
    sample_split = next((s for s in splits if split_role(s) == "train"), splits[0])
    fig_samples(ok, sample_split, figdir / "samples_per_class.png")
    has_dup_fig = fig_duplicates(pairs[pairs["distance"] <= args.threshold],
                                 figdir / "duplicate_examples.png")

    # ---- summary.json
    import PIL
    env = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
           "pillow": PIL.__version__, "imagehash": getattr(imagehash, "__version__", "?"),
           "platform": platform.platform()}
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "root": str(root), "args": {k: str(v) for k, v in vars(args).items()},
        "environment": env, "fingerprint_sha256": fingerprint,
        "image_files": len(df), "readable": total_images, "corrupted": int(len(bad)),
        "non_image_files_ignored": other_files, "num_classes": int(len(counts)),
        "splits": split_tbl.reset_index().to_dict(orient="records"),
        "image_size": size_stats, "formats": formats, "modes": modes,
        "exact_duplicate_groups_md5": n_exact_groups,
        "near_duplicate_sensitivity": sens_tbl.reset_index().to_dict(orient="records"),
        "threshold_used": args.threshold,
        "clusters_at_threshold": {
            "clusters": int(len(clus_df)),
            "images_in_clusters": int(clus_df["size"].sum()) if len(clus_df) else 0,
            "largest_cluster": int(clus_df["size"].max()) if len(clus_df) else 0,
            "cross_split_clusters": int(clus_df["cross_split"].sum()) if len(clus_df) else 0,
            "cross_label_clusters": int(clus_df["cross_label"].sum()) if len(clus_df) else 0,
        },
        "proposed_removals": reason_counts.to_dict(orient="records"),
        "split_sizes_after_removal": after_tbl.reset_index().to_dict(orient="records"),
        "runtime_sec": round(time.time() - t_start, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- summary.md
    cs = summary["clusters_at_threshold"]
    md = [
        "# A2 Dataset EDA - summary",
        "",
        f"- Generated: {summary['generated_at']} by `scripts/a2_eda.py` (seed {SEED})",
        f"- Root: `{root}`",
        f"- Dataset fingerprint (sha256 over sorted `relpath:md5`): `{fingerprint}`",
        f"- Image files: **{len(df)}** (readable {total_images}, corrupted {len(bad)}); "
        f"non-image files ignored: {other_files}",
        f"- Classes: **{len(counts)}**",
        "",
        "## Split overview",
        "",
        md_table(split_tbl),
        "",
        "## Images per class",
        "",
        md_table(counts),
        "",
        "![class distribution](figures/class_distribution.png)",
        "",
        "## Image size and format",
        "",
        f"- Width: min {size_stats['width_min']}, median {size_stats['width_median']}, max {size_stats['width_max']} px",
        f"- Height: min {size_stats['height_min']}, median {size_stats['height_median']}, max {size_stats['height_max']} px",
        f"- Shorter side < 224 px: {size_stats['pct_short_side_lt_224']}% "
        f"(< 128 px: {size_stats['pct_short_side_lt_128']}%)",
        f"- Median aspect ratio (w/h): {size_stats['aspect_ratio_median']}; "
        f"outside [0.5, 2]: {size_stats['pct_aspect_outside_0.5_2']}%",
        f"- Total size on disk: {size_stats['total_size_MB']} MB; median file {size_stats['file_size_median_KB']} KB",
        f"- Formats: {formats}",
        f"- Colour modes: {modes}",
        "",
        "![image sizes](figures/image_sizes.png)",
        "",
        "![samples](figures/samples_per_class.png)",
        "",
        "## Duplicates and leakage",
        "",
        f"- Exact byte-identical groups (MD5): {n_exact_groups}",
        "- Near-duplicates: 64-bit perceptual hash (pHash), Hamming distance threshold sensitivity:",
        "",
        md_table(sens_tbl),
        "",
        f"At threshold **{args.threshold}** (single-linkage clusters): {cs['clusters']} clusters covering "
        f"{cs['images_in_clusters']} images (largest {cs['largest_cluster']}); "
        f"cross-split clusters: **{cs['cross_split_clusters']}**; "
        f"cross-label clusters: **{cs['cross_label_clusters']}**.",
        "",
        "Removal rules: corrupted files dropped; clusters with >1 label dropped entirely (label noise); "
        "cross-split clusters keep only the copy in the most protected split (test > val > train); "
        "within-split copies are optional dedup.",
        "",
        md_table(reason_counts, index=False) if len(reason_counts) else "_No removals proposed._",
        "",
        "### Split sizes after required removals",
        "",
        md_table(after_tbl),
        "",
    ]
    if has_dup_fig:
        md += ["![duplicates](figures/duplicate_examples.png)", "",
               "> pHash can give false positives (e.g. similar plating). Check "
               "`figures/duplicate_examples.png` and `near_duplicate_pairs.csv` before finalising the drop list.", ""]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")

    print("\n" + split_tbl.to_string())
    print(f"\nCorrupted: {len(bad)} | exact dup groups: {n_exact_groups} | "
          f"near-dup clusters @{args.threshold}: {cs['clusters']} "
          f"(cross-split {cs['cross_split_clusters']}, cross-label {cs['cross_label_clusters']})")
    print(f"Done in {summary['runtime_sec']} s -> {out}/summary.md")


if __name__ == "__main__":
    main()
