import torch
from models.final_models.FINAL_hiera_latent_model_head_v25_active import Model
from augmentation.video.base_video_aug import Transformation
import numpy as np
from tqdm import tqdm

from pathlib import Path
from argparse import ArgumentParser


if __name__ == "__main__":
    model_params = {
        "ckpt_dir": None, 
        "backbone_params": {"drop_path_rate": 0.1, "num_cls_tokens": 0},
        "out_dim": 768,
        "mask_ratio": 0.0,
        "init_weights": False,
        "cls_head_name": "models.head_models.dict_head_v6_FINAL_thoughtful",
        "cls_head_params": {
            "latents_params": {"hidden_dim": 512, "out_dim": 384},
            "total_frames": 16,
            "projection_params": {
                # ========================================================================
                # ANGLES
                "full_body_angles": {"hidden_dim": None, "output_dim": 22 * 2},
                "full_right_angles": {"hidden_dim": None, "output_dim": 41 * 2},
                "full_left_angles": {"hidden_dim": None, "output_dim": 41 * 2},
                # ========================================================================
                # KPTS
                "body_pose": {"hidden_dim": None, "output_dim": 61 * 3},
                "right_pose": {"hidden_dim": None, "output_dim": 21 * 3},
                "left_pose": {"hidden_dim": None, "output_dim": 21 * 3},
                # ========================================================================
                # DISTANCE
                "body_dist": {"hidden_dim": None, "output_dim": 12 * 22 * 3},
                "right_dist": {"hidden_dim": None, "output_dim": 5 * 11 * 3},
                "left_dist": {"hidden_dim": None, "output_dim": 5 * 11 * 3},
                # ========================================================================
            },
        },
    }

    transform = Transformation()

    model = Model(**model_params)
    ckpt_dir = (
        "<CHECKPOINT PATH>"
    )
    res = model.load_state_dict(torch.load(ckpt_dir)["model"])
    print(res)
    model.cuda()
    model.eval()

    
    frames = np.zeros((16, 224,224,3))
    frames = transform.aug_video(frames).permute(1, 0, 2, 3)
    frames = frames.unsqueeze(0) # used to create a batch dimension
    # input frames B x 3 x 16 x 224 x 224
    with torch.no_grad():
        y = model(frames.cuda())
        features = y["features"].cpu().numpy()
