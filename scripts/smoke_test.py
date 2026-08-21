"""CPU-only FH-Net construction and forward-pass smoke test."""

from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from model.FH_Net import fhnet


def main():
    torch.manual_seed(1998)
    image_size = 64
    model = fhnet(layers=50, shot=1, pretrained=False)
    model.eval()

    query = torch.randn(1, 3, image_size, image_size)
    support = torch.randn(1, 1, 3, image_size, image_size)
    support_mask = torch.randint(0, 2, (1, 1, image_size, image_size)).float()

    with torch.no_grad():
        logits, prediction = model(x=query, s_x=support, s_y=support_mask)

    expected_logits = (1, 2, image_size, image_size)
    expected_prediction = (image_size, image_size)
    assert tuple(logits.shape) == expected_logits, (tuple(logits.shape), expected_logits)
    assert tuple(prediction.shape) == expected_prediction, (tuple(prediction.shape), expected_prediction)
    assert torch.isfinite(logits).all(), "FH-Net logits contain NaN or Inf."
    assert torch.isfinite(prediction.float()).all(), "FH-Net prediction contains NaN or Inf."
    print("FH-Net CPU smoke test passed.")
    print("logits:", tuple(logits.shape), "prediction:", tuple(prediction.shape))


if __name__ == "__main__":
    main()
