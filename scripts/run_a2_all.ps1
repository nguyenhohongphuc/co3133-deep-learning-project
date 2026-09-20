# A2 - TV 2 - baseline + the two fine-tuning arms of the controlled experiment (handbook 20, 21).
# Run only after the Dataset Proposal is Approved / Approved with conditions (A2-05).
$ErrorActionPreference = "Stop"
foreach ($c in "configs/a2/baseline.yaml", "configs/a2/pretrained_frozen.yaml", "configs/a2/pretrained_full.yaml") {
    Write-Host "=== $c ==="
    python -m src.train --config $c
}
python scripts/make_tables.py --assignment a2
