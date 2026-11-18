# ABO cleaning report

Data and images are credited to Amazon.com under CC BY 4.0. The official listing metadata and image index were downloaded in full. Native images with a maximum axis of 256 pixels were downloaded for the 12 selected product types.

| Stage | Pairs or records |
|---|---:|
| Raw listing records | 147,702 |
| Unique items with English titles and pairing metadata | 121,488 |
| Items in selected class scope before image validation | 19,904 |
| Clean pairs in the selected class scope | 19,230 |
| Selected experiment pairs | 10,627 |

Seed 42. At most 1,250 pairs per class. The split targets 80% train, 10% validation and 10% test, with connected groups kept together.

| Product type | Clean pairs | Train | Validation | Test |
|---|---:|---:|---:|---:|
| BOOT | 909 | 727 | 91 | 91 |
| CHAIR | 1,634 | 1,000 | 125 | 125 |
| FINEEARRING | 724 | 580 | 72 | 72 |
| FINENECKLACEBRACELETANKLET | 939 | 751 | 94 | 94 |
| FINERING | 898 | 718 | 90 | 90 |
| HANDBAG | 681 | 545 | 68 | 68 |
| RUG | 817 | 653 | 82 | 82 |
| SANDAL | 850 | 680 | 85 | 85 |
| SHOES | 9,469 | 1,000 | 125 | 125 |
| SOFA | 969 | 775 | 97 | 97 |
| TABLE | 742 | 594 | 74 | 74 |
| WALL_ART | 598 | 478 | 60 | 60 |

## Cleaning and grouping

Titles are normalized with HTML decoding, Unicode NFKC and whitespace cleanup. Natural product names remain in the text. Images must decode successfully and have a short side of at least 32 pixels. Exact duplicate image/title/label pairs are removed. Identical decoded images with conflicting product labels are excluded. Exclusion reasons and source file/line locations are recorded in data/processed/abo/excluded.csv.

Group links use main-image ID, decoded-pixel hash, identical normalized title, brand/model identifiers of at least five characters, and pHash Hamming distance at most 4. Links are computed before subset selection, including paths through unselected records. The checks found zero cross-split overlap in item IDs, group IDs, image IDs, decoded pixels and normalized titles.

The experiment contains 5,286 groups. The largest group has 1,211 selected pairs.

## Limits

These are automated quality checks. The image/title preview provides a small sample for visual inspection. Shared models and similar hashes can conservatively group distinct products. The native product labels can overlap semantically, and product titles can be strong label cues. Visual inspection found chair cushions under CHAIR and boots under SHOES. The source labels are retained. The selected class scope and cap define a custom subset, so report its exact split when presenting results.

## Reproduction

```bash
python scripts/a3_download_abo.py
python scripts/a3_prepare_abo.py --classes BOOT CHAIR FINEEARRING FINENECKLACEBRACELETANKLET FINERING HANDBAG RUG SANDAL SHOES SOFA TABLE WALL_ART --max-per-class 1250 --seed 42 --phash-distance 4 --download-images --workers 16
```

Run the same preparation command without --download-images to reproduce cleaning offline. See summary.json for source URLs, archive/index SHA-256 values, package versions and actual counts. The individual-image source ledger is data/raw/abo/image_sources.jsonl.
