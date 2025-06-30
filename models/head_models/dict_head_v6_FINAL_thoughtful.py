import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        if hidden_dim is None:
            self.layers = nn.Sequential(
                nn.Linear(input_dim, output_dim),
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),  # Output keypoints
            )

    def forward(self, x):
        x = self.layers(x)

        return x


class PermuteLayer(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


class ProjectionModel(nn.Module):
    def __init__(
        self,
        latents_params,
        projection_params,
        total_frames,
        in_dim,
    ):
        super().__init__()

        hidden_latent = latents_params["out_dim"]
        self.latent_proj = torch.nn.Sequential(
            torch.nn.Conv1d(in_dim, latents_params["hidden_dim"], 1, bias=True),
            torch.nn.GELU(),
            torch.nn.ConvTranspose1d(
                latents_params["hidden_dim"],
                hidden_latent,
                total_frames,
                bias=True,
            ),
            # torch.nn.Conv1d(latents_params["hidden_dim"], hidden_latent, 1, bias=True),
            # torch.nn.BatchNorm1d(hidden_latent),
            # torch.nn.GELU(),
            PermuteLayer((0, 2, 1)),
        )
        dict_component_projections = {}
        projection_params = projection_params
        for key, pp_params in projection_params.items():
            dict_component_projections[key] = MLP(input_dim=hidden_latent, **pp_params)

        self.dict_component_projections = nn.ModuleDict(dict_component_projections)

    def forward(self, x):
        dict_results = {}

        latent = x
        dict_results["latent"] = latent
        post_latent = self.latent_proj(latent.unsqueeze(-1))
        dict_results["post_latent"] = post_latent
        for key, proj in self.dict_component_projections.items():
            dict_results[key] = proj(post_latent)

        return dict_results
