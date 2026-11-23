# 16-825 Assignment 3: Volume Rendering, NeRF, and 3D Gaussian Splatting

Andrew ID: `skorada`

This package contains the source code for the volume-rendering, NeRF, VolSDF,
and 3D Gaussian Splatting experiments. Generated results and the full write-up
are provided separately in the `proj3` webpage submission.

## Environment

Install the Python dependencies from the package root:

```bash
pip install -r requirements.txt
```

PyTorch and PyTorch3D must be installed with versions compatible with the
machine's CUDA installation. The original Conda specification for the NeRF
portion is included as `nerf/environment.yml`.

## Data

The course-provided datasets are intentionally excluded from the code archive.
Place the NeRF data under `nerf/data/`, the pretrained Gaussian file at
`GS/data/chair.ply`, and the truck dataset under `GS/data/truck/`. The materials
dataset is required for the view-dependent NeRF experiment.

## Running experiments

`main.py` provides a single entry point:

```bash
python main.py volume-rendering
python main.py optimize-box
python main.py nerf-lego
python main.py nerf-materials
python main.py volsdf
python main.py volsdf-ablation
python main.py gaussian-test
python main.py gaussian-render
python main.py gaussian-train --num_itrs 1000
```

Arguments after an experiment name are passed to its underlying program. Hydra
overrides can therefore be appended to the NeRF commands, while standard flags
can be appended to the Gaussian commands.

The spherical-harmonic extension is implemented in
`GS/data_utils.py::colours_from_spherical_harmonics` and
`GS/model.py::calculate_gaussian_directions`. The marked Q7.1 switches in
`GS/model.py` select spherical-harmonic color instead of DC-only color when
rendering the extension.

## Main implementation files

- `nerf/ray_utils.py`: pixels, rays, and ray bundles
- `nerf/sampler.py`: stratified point sampling
- `nerf/renderer.py`: volume and VolSDF rendering
- `nerf/implicit.py`: NeRF and neural SDF models
- `nerf/losses.py`: Eikonal loss
- `GS/model.py`: Gaussian projection, sorting, and rasterization
- `GS/data_utils.py`: data loading and spherical harmonics
- `GS/train.py`: Gaussian optimization and evaluation
