# A1-14 - TV 2 - the main comparison repeated over 3 seeds -> mean +/- std (handbook 3.3).
$ErrorActionPreference = "Stop"
$configs = @(
    "configs/a1/linear.yaml",
    "configs/a1/mlp.yaml",
    "configs/a1/cnn.yaml",
    "configs/a1/rnn_rows.yaml",
    "configs/a1/transformer.yaml"
)
foreach ($seed in 42, 1, 2) {
    foreach ($c in $configs) {
        Write-Host "=== $c (seed $seed) ==="
        python -m src.train --config $c --seed $seed
    }
}
python scripts/make_tables.py --assignment a1
