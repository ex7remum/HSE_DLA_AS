import logging
from typing import List
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    all_audios = [item['audio'] for item in dataset_items]
    batch_audio = torch.stack(all_audios)

    batch_label = torch.tensor([item['label'] for item in dataset_items])

    batch_attack = [item['attack'] for item in dataset_items]

    return {
        "audios": batch_audio,
        "labels": batch_label,
        "attack": batch_attack
    }
