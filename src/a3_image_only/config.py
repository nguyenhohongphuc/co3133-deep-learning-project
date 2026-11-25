from pathlib import Path
from dataclasses import dataclass

REPO = Path(__file__).resolve().parents[2]
@dataclass
class Config:
    image_root: Path = Path(r'D:\HK261\DeepLearning\Assignment\Assignment3\Data\images\small')
    split_dir: Path | None = None  # set explicitly if auto-discovery finds multiple directories
    output_dir: Path = REPO / 'results' / 'a3_image_only' / 'resnet50'
    backbone: str = 'resnet50'
    image_size: int = 224
    batch_size: int = 32
    workers: int = 0  # Windows/Jupyter safe; scripts can use 4
    epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.2
    seed: int = 42
    amp: bool = True
    def splits(self):
        if self.split_dir is not None:
            return Path(self.split_dir)
        names = ('train.csv','validation.csv','test.csv','labels.json')
        candidates = [p.parent for p in (REPO/'data').rglob('labels.json') if all((p.parent/n).is_file() for n in names)]
        if len(candidates) != 1:
            raise RuntimeError(f'Set cfg.split_dir explicitly. Candidates: {candidates}')
        return candidates[0]
