
# SignRep: Enhancing Self-Supervised Sign Representations

SignRep contains research code and models for "SignRep: Enhancing Self-Supervised Sign Representations".

## Contents

- `models/` - Model implementations and final model definitions used for evaluation and feature extraction.
- `head_models/` - projection head implementations used during pretraining.
- `augmentation/video/` - Video augmentation utilities and transformations.
- `example_usage.py` - Minimal script showing how to load a checkpoint and extract features from a video.

## Quick features

- Pretrained checkpoints (see release tags on the repository) for extracting representations.
- Modular model + head design so you can swap backbone or head implementations.
- Video augmentation utilities for preprocessing input videos into model-ready tensors.

## Requirements

We provide a minimal `requirements.txt` with the dependencies used by the example script. 
Install using a virtual environment (example):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

See `example_usage.py` for a compact example that:

- loads a model definition from `models.final_models.FINAL_hiera_latent_model_head_v25_active.Model`
- applies `augmentation.video.base_video_aug.Transformation` to raw video frames
- iterates over video segments and extracts `features` and `latent` outputs from the model

To run the example, edit the `ckpt_dir` and `video_dir` variables in `example_usage.py` (or adapt it to accept CLI args), then run:

```bash
python example_usage.py
```

## Checkpoints

Model checkpoints are published in the repository release tags. Download the checkpoint appropriate for the model architecture you want to use and point `example_usage.py` (or your own script) at the checkpoint path.

## Citation

If you use SignRep in your research, please cite the associated paper:

```cite
@article{wong2025signrep,
  title={Signrep: Enhancing self-supervised sign representations},
  author={Wong, Ryan and Camgoz, Necati Cihan and Bowden, Richard},
  journal={arXiv preprint arXiv:2503.08529},
  year={2025}
}
```


## License

This project is distributed under the terms of the included `LICENSE` file.

## Contact

If you have questions, open an issue in this repository.
