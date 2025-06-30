import torch
import torch.nn as nn
import importlib
import torch.nn.functional as F
from models.hiera.hiera_nodec_mae import mae_hiera_base_16x224


class Model(nn.Module):
    def __init__(
        self,
        backbone_params,
        ckpt_dir,
        mask_ratio,
        out_dim,
        cls_head_name,
        cls_head_params,
        init_weights=True,
    ):
        super(Model, self).__init__()
        self.backbone = mae_hiera_base_16x224(
            num_classes=400, pretrained=False, **backbone_params
        )
        if ckpt_dir is not None:
            ckpt = torch.load(ckpt_dir, map_location="cpu")

            res = self.backbone.load_state_dict(ckpt["model_state"], strict=False)
            print(res)
        self.mask_ratio = mask_ratio
        self.num_features = out_dim

        try:
            self.cls_head = importlib.import_module(cls_head_name).ProjectionModel(
                **cls_head_params, in_dim=self.num_features
            )

            self.fc = torch.nn.Sequential(
                nn.LayerNorm(out_dim),
                nn.Linear(out_dim, out_dim, bias=False),
            )

            self.active_head = torch.nn.Sequential(
                torch.nn.Linear(out_dim, 128),
                torch.nn.GELU(),
                torch.nn.Linear(128, 2),
                # torch.nn.Sigmoid(),
            )

            if init_weights:
                self.cls_head.apply(self._init_weights)
        except:
            self.cls_head = nn.Identity()
            print("SKIPPED CLASS HEAD")

    def forward(self, imgs, extract_features=False, apply_results=False):
        b, c, t, h, w = imgs.shape

        # if extract_features:
        #     return {"features": self.backbone(imgs, eval=True)}
        latent, intermediates_feature_map, intermediate_features_cls = self.backbone(
            imgs,
            mask_ratio=self.mask_ratio if self.training else 0.0,
            # eval=not self.training,
        )

        res = self.fc(latent)

        cls_y = self.cls_head(res)
        cls_active = self.active_head(res)

        return {
            "intermediates_feature_map": intermediates_feature_map,
            "intermediate_features_cls": intermediate_features_cls,
            "features": res,
            "latent": latent,
            # "post_features": res,
            "cls_head": cls_y,
            "cls_active": cls_active,
        }
