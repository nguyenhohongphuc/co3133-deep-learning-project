# Assignment 3 — Image-only

Uses the existing team manifests unchanged (12 classes). No text is fed to the model. Default ResNet-50 ImageNet fine-tuning, white letterbox, 224 pixels, FP16 on CUDA. Confirm backbone and preprocessing with fusion team before claiming a controlled comparison.

## Setup
Place these files in `src/a3_image_only/` in the repository. In the existing `co3133` environment, install missing packages using `pip install pandas numpy pillow matplotlib scikit-learn tqdm ipykernel`. Requires compatible PyTorch/torchvision with CUDA and Python 3.10+, recent PyTorch with `torch.amp.GradScaler`. Do not replace your working CUDA build blindly. First pretrained run requires Internet or cached weights.

Set `Config.image_root` if needed. `Config.split_dir` auto-discovers the directory containing all four official files under repo/data; if ambiguous specify it explicitly. Never use the independently generated 33-class subset.

Run notebooks in order, or from this directory run `python train.py --workers 4` then `python evaluate.py`. Resume interrupted training with `python train.py --workers 4 --resume` using identical config; checkpoints are local trusted files only. Notebook default workers=0 for Windows safety. For a new experiment use a separate output directory. Best checkpoint selected solely by validation macro-F1; test only after configuration is finalized. Results saved under repo/results/a3_image_only/resnet50. Do not commit raw data or large checkpoints; provide checkpoint access or reconstruction instructions.

Metrics include listing-level accuracy and macro-F1, per-class metrics, confusion matrix, predictions and errors, wall-clock epoch time, isolated forward latency and parameter count. Repeated images within a split are retained to preserve the team protocol. Exact-overlap checks are not proof against all near duplicates.

Data attribution: Amazon.com; dataset construction by Matthieu Guillaumin, Thomas Dideriksen, Kenan Deng, Himanshu Arora, Jasmine Collins and Jitendra Malik. CC BY 4.0 https://creativecommons.org/licenses/by/4.0/. Images are resized/padded and augmented for training. See https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html.

AI disclosure: ChatGPT (GPT-6.2) assisted code and experiment design on 2026-11-24. Student must verify local execution, record representative prompts, modifications and verification evidence, and add an entry to the team's AI_USAGE.md and report. No training results are supplied in this package.
