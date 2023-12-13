import torch
import torch.nn as nn


class RawNetLoss(nn.Module):
    def __init__(self):
        super(RawNetLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 9.0]))  # bona-fide is 1, spoof is 0

    def forward(self, logits, labels, **kwargs):
        return {
            "loss": self.ce(logits, labels)
        }
