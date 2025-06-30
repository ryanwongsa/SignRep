import random
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class Transformation(object):
    def __init__(
        self,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ):
        self.transform_valid = A.Compose(
            [
                A.Resize(224, 224, p=1.0),
                A.Normalize(
                    mean=mean,
                    std=std,
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(p=1.0),
            ],
            additional_targets=None,
        )

    def aug_video(self, frames):
        image_keys = ["image"]
        additional_targets = {}
        for i in range(1, len(frames)):
            additional_targets[f"image_{i}"] = "image"
            image_keys.append(f"image_{i}")

        sample = {"image": frames[0]}
        for i in range(1, len(frames)):
            sample[f"image_{i}"] = frames[i]

        self.transform_valid.add_targets(additional_targets)
        sample = self.transform_valid(**sample)

        return torch.stack([sample[ik] for ik in image_keys])
