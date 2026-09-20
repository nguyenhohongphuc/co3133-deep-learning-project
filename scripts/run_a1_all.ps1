# A1-13 - TV 2 - all five models, one split, one seed (handbook 12.2).
# Usage:  .\scripts\run_a1_all.ps1
$ErrorActionPreference = "Stop"
$configs = @(
    "configs/a1/linear.yaml",
    "configs/a1/mlp.yaml",
    "configs/a1/cnn.yaml",
    "configs/a1/rnn_rows.yaml",
    "configs/a1/transformer.yaml"
)
foreach ($c in $configs) {
    Write-Host "=== $c ==="
    python -m src.train --config $c
}
python scripts/make_tables.py --assignment a1
