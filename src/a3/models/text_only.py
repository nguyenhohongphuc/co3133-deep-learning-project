import torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    def __init__(self, model_name, revision=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, revision=revision)
        self.out_dim = self.bert.config.dim              # 768

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B, L, 768]
        m = attention_mask.unsqueeze(-1).float()                # [B, L, 1]
        return (out * m).sum(1) / m.sum(1).clamp(min=1e-6)      # [B, 768]
    def freeze(self):
        """Đóng băng toàn bộ DistilBERT."""
        for p in self.bert.parameters():
            p.requires_grad = False

    def unfreeze_last(self, n):
        layers = self.bert.transformer.layer
        if not 0 <= n <= len(layers):
            raise ValueError(f"n phải nằm trong [0, {len(layers)}], nhận được {n}")
        for layer in layers[len(layers) - n:]:      # n=0 -> layers[6:] = rỗng
            for p in layer.parameters():
                p.requires_grad = True
                
class TextClassifier(nn.Module):
    def __init__(self, model_name, num_classes=12, dropout=0.1, revision=None):
        super().__init__()
        self.encoder = TextEncoder(model_name, revision=revision)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.encoder.out_dim, num_classes))

    def forward(self, batch):
        h = self.encoder(batch["input_ids"], batch["attention_mask"])   # [B, 768]
        return self.head(h)           # [B, 12]
    def freeze_encoder(self):
        self.encoder.freeze()

    def unfreeze_last(self, n):
        self.encoder.unfreeze_last(n)