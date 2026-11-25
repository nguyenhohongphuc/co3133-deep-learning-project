"""All fusion checks, including real-pair integration, in one test suite."""
import unittest
import copy
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from transformers import BertTokenizerFast, DistilBertConfig

from src.a3.fusion_data import image_transform
from src.a3.fusion_engine import classification_metrics
from src.a3.models.fusion import FusionClassifier


def offline_tokenizer(folder):
    """Tiny local vocabulary: tests never download tokenizer or model weights."""
    path = Path(folder) / "vocab.txt"
    path.write_text("\n".join([
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "shoe", "sandal", "boot",
        "chair", "table", "ring", "amazon", "collection", "diamond", "stud", "earrings",
        "platinum", "round", "cut", "certified", "princess", "gold", "silver", "small",
        "large", "red", "blue", "black", "white", "and", "the", "a",
    ]) + "\n", encoding="utf-8")
    return BertTokenizerFast(vocab_file=str(path), do_lower_case=True)


class FusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.model = FusionClassifier(image_weights=None, text_config=DistilBertConfig(
            vocab_size=32, dim=768, n_layers=2, n_heads=12, hidden_dim=48,
            dropout=0, attention_dropout=0, max_position_embeddings=128).to_dict())

    def test_masked_mean_ignores_extra_padding(self):
        encoder = self.model.text_encoder.eval()
        with torch.no_grad():
            a = encoder(torch.tensor([[2, 5, 3]]), torch.tensor([[1, 1, 1]]))
            b = encoder(torch.tensor([[2, 5, 3, 0, 0]]), torch.tensor([[1, 1, 1, 0, 0]]))
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)

    def test_controlled_unfreezing_and_eval_mode(self):
        model = self.model
        model.configure_finetuning(False, 0)
        model.train()
        self.assertFalse(model.image_encoder.bn1.training)
        self.assertFalse(model.text_encoder.bert.training)
        model.configure_finetuning(True, 1)
        self.assertTrue(model.image_encoder.layer4.training)
        self.assertFalse(model.image_encoder.layer3.training)
        self.assertTrue(model.text_encoder.bert.transformer.layer[-1].training)
        self.assertFalse(model.text_encoder.bert.transformer.layer[0].training)
        model.eval()
        self.assertFalse(model.image_encoder.layer4.training)
        self.assertFalse(model.text_encoder.bert.transformer.layer[-1].training)
        with self.assertRaises(ValueError):
            model.configure_finetuning(True, 3)

    def test_both_modalities_affect_logits(self):
        model = self.model.eval()
        batch = {"images": torch.randn(2, 3, 64, 64),
                 "input_ids": torch.tensor([[2, 5, 3], [2, 6, 3]]),
                 "attention_mask": torch.ones(2, 3, dtype=torch.long)}
        with torch.no_grad():
            base = model(batch)
            image_changed = model({**batch, "images": torch.zeros_like(batch["images"])})
            text_changed = model({**batch, "input_ids": torch.tensor([[2, 7, 3], [2, 8, 3]])})
        self.assertEqual(tuple(base.shape), (2, 12))
        self.assertFalse(torch.allclose(base, image_changed))
        self.assertFalse(torch.allclose(base, text_changed))

    def test_letterbox_keeps_entire_product_and_white_padding(self):
        tensor = image_transform(False)(Image.new("RGB", (200, 100), "red"))
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        torch.testing.assert_close(tensor[:, 0, 0], (torch.ones(3) - mean) / std)
        torch.testing.assert_close(tensor[:, 112, 112], (torch.tensor([1., 0., 0.]) - mean) / std)

    def test_macro_f1_includes_absent_classes(self):
        cm = torch.zeros(12, 12, dtype=torch.long)
        cm[0, 0], cm[1, 0], cm[1, 1] = 2, 1, 1
        metrics = classification_metrics(cm)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["macro_f1"], (0.8 + 2 / 3) / 12)
        self.assertEqual(metrics["per_class"][2]["f1"], 0)

    def test_training_driver_and_offline_evaluation(self):
        from scripts import a3_train_fusion, a3_evaluate_fusion
        from src.a3.fusion_data import load_fusion_splits, make_collator, make_dataset
        from src.a3.fusion_engine import load_checkpoint, read_config

        cfg = copy.deepcopy(read_config("configs/a3_fusion.yaml"))
        cfg["train"].update(epochs=2, freeze_epochs=1, unfreeze_last=1, batch_size=2)
        cfg["data"]["workers"] = 0
        splits, labels = load_fusion_splits(cfg, check_images=False)
        # Audit the full manifests first; shorten only the integration run, not the real trainer.
        mini = {name: rows[:2] for name, rows in splits.items()}
        with tempfile.TemporaryDirectory(prefix="a3-fusion-driver-") as folder:
            tokenizer = offline_tokenizer(folder)
            dataset = make_dataset(cfg, mini["train"], train=False)
            batch = make_collator(cfg, tokenizer)([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch["images"].shape), (2, 3, 224, 224))
            self.assertLessEqual(batch["input_ids"].shape[1], 96)
            self.model.eval()
            with torch.no_grad():
                image, text = self.model.encode_modalities(batch)
            self.assertEqual(tuple(image.shape), (2, 2048))
            self.assertEqual(tuple(text.shape), (2, 768))

            real_run_epoch = a3_train_fusion.run_epoch
            stages, validation_logits = [], []

            def checked_epoch(model, loader, device, **kwargs):
                training = kwargs.get("optimizer") is not None
                stem_bn = model.image_encoder.bn1.running_mean.clone()
                head_before = model.head[0].weight.detach().clone()
                result = real_run_epoch(model, loader, device, **kwargs)
                if training:
                    self.assertFalse(torch.equal(head_before, model.head[0].weight))
                    torch.testing.assert_close(model.image_encoder.bn1.running_mean, stem_bn, rtol=0, atol=0)
                    fine_tune = model.image_encoder.layer4[0].conv1.weight.requires_grad
                    stages.append(fine_tune)
                    if fine_tune:
                        self.assertIsNotNone(model.image_encoder.layer4[0].conv1.weight.grad)
                        self.assertIsNone(model.image_encoder.conv1.weight.grad)
                        layers = model.text_encoder.bert.transformer.layer
                        self.assertIsNotNone(next(layers[-1].parameters()).grad)
                        self.assertIsNone(next(layers[0].parameters()).grad)
                    else:
                        self.assertTrue(all(p.grad is None for p in model.image_encoder.parameters()))
                        self.assertTrue(all(p.grad is None for p in model.text_encoder.parameters()))
                else:
                    with torch.no_grad():
                        validation_logits.append(model(batch).clone())
                return result

            cfg["output_dir"] = str(Path(folder) / "run")
            with patch.object(a3_train_fusion, "load_fusion_splits", return_value=(mini, labels)), \
                 patch.object(a3_train_fusion, "build_fusion", return_value=self.model), \
                 patch.object(a3_train_fusion, "run_epoch", side_effect=checked_epoch), \
                 patch.object(a3_train_fusion.AutoTokenizer, "from_pretrained", return_value=tokenizer):
                out = a3_train_fusion.train(cfg, "cpu")
            self.assertEqual(stages, [False, True])
            self.assertTrue((out / "best.pt").is_file())
            self.assertTrue((out / "requirements.lock.txt").is_file())
            restored, checkpoint = load_checkpoint(out / "best.pt")
            restored.eval()
            with torch.no_grad():
                logits = restored(batch)
            self.assertEqual(tuple(logits.shape), (2, 12))
            torch.testing.assert_close(logits, validation_logits[checkpoint["epoch"] - 1], rtol=0, atol=0)
            # Here tokenizer loading is real, local-only; no Hub calls are mocked.
            with patch.object(a3_evaluate_fusion, "load_fusion_splits", return_value=(mini, labels)):
                metrics = a3_evaluate_fusion.evaluate(out / "best.pt", device_name="cpu")
            self.assertEqual(metrics["samples"], 2)
            self.assertTrue((out / "evaluation_best_validation" / "predictions.csv").is_file())


if __name__ == "__main__":
    unittest.main()
