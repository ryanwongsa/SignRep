import torch
from models.final_models.FINAL_hiera_latent_model_head_v25_active import Model
from augmentation.video.base_video_aug import Transformation
import numpy as np
from tqdm import tqdm

from pathlib import Path
from argparse import ArgumentParser
import cv2


if __name__ == "__main__":
    model_params = {
        "ckpt_dir": None,
        "backbone_params": {"drop_path_rate": 0.1, "num_cls_tokens": 0},
        "out_dim": 768,
        "mask_ratio": 0.0,
        "init_weights": False,
        "cls_head_name": "models.head_models.dict_head_v6_FINAL_thoughtful", # head to predict sign priors
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

    ckpt_dir = "<ckpt_dir>"
    video_dir = "<video_dir>"
    stride = 2 
    segment_length = 16 
    
    transform = Transformation()

    model = Model(**model_params)

    res = model.load_state_dict(torch.load(ckpt_dir)["model"])
    print(res)
    model.cuda()
    model.eval()

    video_path = Path(video_dir)
    cap = cv2.VideoCapture(str(video_path))
    original_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_frames.append(frame)

    frames = transform.aug_video(original_frames) # T x 3 x 224 x 224
    list_of_features = []
    list_of_latents = []
    for i in range(0, frames.shape[0], stride):
        sel_frames = frames[i:i+segment_length,:, :, :]
        if sel_frames.shape[0] < segment_length:
            continue
        sel_frames = sel_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)  # B x 3 x 16 x 224 x 224

        with torch.no_grad():
            y = model(sel_frames.cuda())
            features = y["features"][0].cpu().numpy()
            latents = y['latent'][0].cpu().numpy()
            list_of_features.append(features)
            list_of_latents.append(latents)

