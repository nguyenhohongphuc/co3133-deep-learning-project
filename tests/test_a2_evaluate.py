"""Offline evaluator checks using synthetic images and temporary checkpoints.

    python -m unittest discover -s tests -p 'test_a2_evaluate.py' -v
"""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image
import torch
import yaml

from src.a2.evaluate import evaluate_run, parse_args
from src.a2.models import ResNet50, SmallCNN
from torchvision import models as tv_models


class A2EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="a2-evaluator-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.data_dir = self.root / "images"
        self.classes = ["class_0", "class_1", "class_2", "class_3"]
        self.split = dict(classes=self.classes, train=[], val=[], test=[])
        for index, label in enumerate([0, 0, 1, 1, 2]):
            self.split["val"].append(self.make_image("val", index, label))
        self.split["test"].append(self.make_image("test", 0, 2))
        self.split["counts"] = {key: len(self.split[key]) for key in ("train", "val", "test")}
        self.split_file = self.root / "split.json"
        self.split_file.write_text(json.dumps(self.split))
        self.config = dict(
            seed=42, model="cnn", data_dir=str(self.data_dir), split_file=str(self.split_file),
            data=dict(image_size=8, resize=10, crop_scale=[0.08, 1.0], hflip=True,
                      color_jitter=0.1, stats={"train": dict(mean=[0, 0, 0], std=[1, 1, 1]),
                                             "imagenet": dict(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])}),
            training=dict(batch_size=2, device="cuda", epochs=5),
            models={"cnn": dict(normalize="train", params=dict(channels=[2], num_classes=4, dropout=0.3),
                                training=dict(batch_size=99))},
        )
        model = SmallCNN(**self.config["models"]["cnn"]["params"])
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.classifier[-1].bias.copy_(torch.tensor([3., 2., 1., 0.]))
        torch.save(model.state_dict(), self.run_dir / "best.pt")
        self.save_config()

    def make_image(self, split, index, label):
        relative = f"{split}/{self.classes[label]}/{index}.png"
        target = self.data_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12 + index, 10 + index), color=(30 * index, 90, 170)).save(target)
        return dict(path=relative, label=label, md5=hashlib.md5(target.read_bytes()).hexdigest())

    def save_config(self):
        (self.run_dir / "config.yaml").write_text(yaml.safe_dump(self.config))
        (self.run_dir / "metrics.json").write_text(json.dumps(
            dict(model=self.config["model"], run_id="synthetic_run", commit="synthetic-training-commit")))

    def args(self, *extra):
        return parse_args(["--run-dir", str(self.run_dir), "--device", "cpu", *extra])

    @staticmethod
    def read_csv(path):
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_predictions_metrics_and_metadata_match_known_logits(self):
        result, output = evaluate_run(self.args())
        rows = self.read_csv(output / "predictions.csv")
        self.assertEqual([row["path"] for row in rows], [item["path"] for item in self.split["val"]])
        self.assertEqual([int(row["index"]) for row in rows], list(range(5)))
        self.assertEqual([int(row["pred_label"]) for row in rows], [0] * 5)
        self.assertEqual([int(row["correct"]) for row in rows], [1, 1, 0, 0, 0])
        probabilities = torch.tensor([3., 2., 1., 0.]).softmax(0)
        for index, row in enumerate(rows):
            self.assertEqual(row["true_class"], self.classes[self.split["val"][index]["label"]])
            self.assertEqual(row["md5"], self.split["val"][index]["md5"])
            self.assertEqual(int(row["width"]), 12 + index)
            self.assertEqual(int(row["height"]), 10 + index)
            self.assertAlmostEqual(float(row["aspect_ratio"]), (12 + index) / (10 + index))
            self.assertAlmostEqual(float(row["confidence"]), float(probabilities[0]))
            self.assertAlmostEqual(float(row["confidence_margin"]), float(probabilities[0] - probabilities[1]))
            self.assertAlmostEqual(float(row["true_class_probability"]),
                                   float(probabilities[int(row["true_label"])]))
            self.assertEqual([int(row["top2_label"]), int(row["top3_label"])], [1, 2])
        self.assertAlmostEqual(result["accuracy"], 2 / 5)
        self.assertAlmostEqual(result["macro_f1"], 1 / 7)
        self.assertAlmostEqual(result["weighted_f1"], 8 / 35)
        self.assertEqual(result["top3_accuracy"], 1.0)
        expected_loss = -sum(float(probabilities[label].log()) for label in [0, 0, 1, 1, 2]) / 5
        self.assertAlmostEqual(result["loss"], expected_loss, places=6)
        self.assertEqual(result["confusion"], [[2, 0, 0, 0], [2, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(result["batch_size"], 2)  # Saved resolved config wins over model defaults.
        self.assertEqual(result["device"], "cpu")  # Training config requests CUDA.
        self.assertEqual(result["training_commit"], "synthetic-training-commit")
        self.assertEqual(result["checkpoint_sha256"], hashlib.sha256((self.run_dir / "best.pt").read_bytes()).hexdigest())
        per_class = self.read_csv(output / "per_class_metrics.csv")
        self.assertEqual(int(per_class[3]["support"]), 0)
        self.assertEqual(float(per_class[3]["f1"]), 0.0)
        self.assertEqual(json.loads((output / "metrics.json").read_text())["num_samples"], 5)
        with (output / "confusion_matrix.csv").open(newline="") as stream:
            self.assertEqual(next(csv.reader(stream))[1:], self.classes)

    def test_split_selection_overrides_and_repeatability(self):
        _, val_dir = evaluate_run(self.args())
        original = (val_dir / "predictions.csv").read_bytes()
        evaluate_run(self.args())
        self.assertEqual(original, (val_dir / "predictions.csv").read_bytes())
        self.config["data_dir"] = "missing-dataset"
        self.config["split_file"] = "missing-split.json"
        self.save_config()
        result, test_dir = evaluate_run(self.args("--split", "test", "--data-dir", str(self.data_dir),
                                                  "--split-file", str(self.split_file), "--batch-size", "1"))
        self.assertEqual(result["num_samples"], 1)
        self.assertEqual(result["split"], "test")
        self.assertEqual(result["batch_size"], 1)
        self.assertEqual(self.read_csv(test_dir / "predictions.csv")[0]["path"], self.split["test"][0]["path"])
        self.assertEqual(original, (val_dir / "predictions.csv").read_bytes())

    def test_missing_or_incompatible_checkpoint_does_not_publish_results(self):
        with self.assertRaisesRegex(FileNotFoundError, "Checkpoint not found"):
            evaluate_run(self.args("--checkpoint", str(self.root / "missing.pt")))
        self.assertFalse((self.run_dir / "evaluation").exists())
        torch.save(SmallCNN(channels=[2], num_classes=3).state_dict(), self.root / "wrong.pt")
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            evaluate_run(self.args("--checkpoint", str(self.root / "wrong.pt")))
        self.assertFalse((self.run_dir / "evaluation").exists())

    def test_empty_split_and_output_collision_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "preserve training metrics"):
            evaluate_run(self.args("--output-dir", str(self.run_dir)))
        self.split["val"] = []
        self.split["counts"]["val"] = 0
        self.split_file.write_text(json.dumps(self.split))
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluate_run(self.args())

    def test_both_resnet_modes_load_offline_from_checkpoint(self):
        model = ResNet50(num_classes=4, pretrained=False)
        with torch.no_grad():
            model.net.fc.weight.zero_()
            model.net.fc.bias.copy_(torch.tensor([3., 2., 1., 0.]))
        checkpoint = self.root / "resnet.pt"
        torch.save(model.state_dict(), checkpoint)
        del model
        for name, frozen in [("resnet50_frozen", True), ("resnet50_full", False)]:
            with self.subTest(model=name):
                self.config["model"] = name
                self.config["models"] = {name: dict(normalize="imagenet",
                                                    params=dict(num_classes=4, freeze_backbone=frozen))}
                self.save_config()
                with mock.patch.object(tv_models.ResNet50_Weights, "get_state_dict",
                                       side_effect=AssertionError("Must not download pretrained weights")):
                    result, _ = evaluate_run(self.args("--checkpoint", str(checkpoint),
                                                       "--output-dir", str(self.root / name)))
                self.assertAlmostEqual(result["accuracy"], 2 / 5)
                self.assertAlmostEqual(result["macro_f1"], 1 / 7)


if __name__ == "__main__":
    unittest.main()
