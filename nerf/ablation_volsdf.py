import json
import math
from pathlib import Path

import mcubes
import numpy as np
import torch
from omegaconf import OmegaConf

from dataset import get_nerf_datasets
from losses import get_eikonal_loss
from ray_utils import (
    get_random_pixels_from_image,
    get_rays_from_pixels,
    sample_images_at_xy,
)
from volume_rendering_main import Model


VARIANTS = {
    "baseline": {"eikonal_weight": 0.02, "alpha": 10.0, "beta": 0.05},
    "eikonal_low": {"eikonal_weight": 0.005, "alpha": 10.0, "beta": 0.05},
    "eikonal_high": {"eikonal_weight": 0.08, "alpha": 10.0, "beta": 0.05},
    "alpha_low": {"eikonal_weight": 0.02, "alpha": 5.0, "beta": 0.05},
    "alpha_high": {"eikonal_weight": 0.02, "alpha": 20.0, "beta": 0.05},
    "beta_low": {"eikonal_weight": 0.02, "alpha": 10.0, "beta": 0.02},
    "beta_high": {"eikonal_weight": 0.02, "alpha": 10.0, "beta": 0.10},
}

EPOCHS = 25
GRID_SIZE = 64
RESULT_DIR = Path("ablation_results")


@torch.no_grad()
def validation_metrics(model, val_dataset, image_size):
    squared_error = 0.0
    value_count = 0
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(12345)
    for entry in val_dataset:
        image = entry["image"].cuda().unsqueeze(0)
        camera = entry["camera"].cuda()
        xy = get_random_pixels_from_image(4096, image_size, camera)
        rays = get_rays_from_pixels(xy, image_size, camera)
        prediction = model(rays)["feature"]
        target = sample_images_at_xy(image, xy)
        squared_error += torch.sum((prediction - target) ** 2).item()
        value_count += target.numel()
    torch.random.set_rng_state(generator_state)
    mse = squared_error / value_count
    return {"val_mse": mse, "val_psnr": -10.0 * math.log10(max(mse, 1e-12))}


@torch.no_grad()
def mesh_metrics(implicit_fn):
    axis = torch.linspace(-3.0, 3.0, GRID_SIZE + 1, device="cuda")
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    distances = []
    flat_grid = grid.reshape(-1, 3)
    for start in range(0, flat_grid.shape[0], 65536):
        distances.append(implicit_fn.get_distance(flat_grid[start : start + 65536]))
    sdf = torch.cat(distances).reshape(GRID_SIZE + 1, GRID_SIZE + 1, GRID_SIZE + 1)
    vertices, triangles = mcubes.marching_cubes(sdf.cpu().numpy(), 0.0)
    if len(vertices) == 0:
        return {"mesh_vertices": 0, "mesh_faces": 0, "horizontal_planarity": 0.0}

    vertices = (vertices / GRID_SIZE - 0.5) * 6.0
    hist, _ = np.histogram(vertices[:, 2], bins=GRID_SIZE, range=(-3.0, 3.0))
    horizontal_planarity = float(hist.max() / len(vertices))
    return {
        "mesh_vertices": int(len(vertices)),
        "mesh_faces": int(len(triangles)),
        "horizontal_planarity": horizontal_planarity,
        "sdf_min": float(sdf.min()),
        "sdf_max": float(sdf.max()),
    }


def main():
    RESULT_DIR.mkdir(exist_ok=True)
    base_cfg = OmegaConf.load("configs/volsdf_surface.yaml")
    train_dataset, val_dataset, _ = get_nerf_datasets(
        base_cfg.data.dataset_name,
        [base_cfg.data.image_size[1], base_cfg.data.image_size[0]],
        autodownload=False,
    )

    all_results = {}
    for name, params in VARIANTS.items():
        print(f"ABLATION_START {name} {params}", flush=True)
        torch.manual_seed(base_cfg.seed)
        np.random.seed(base_cfg.seed)
        cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        cfg.training.eikonal_weight = params["eikonal_weight"]
        cfg.renderer.alpha = params["alpha"]
        cfg.renderer.beta = params["beta"]

        model = Model(cfg).cuda().train()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
        final_rgb = 0.0
        final_eikonal = 0.0

        for epoch in range(EPOCHS):
            rgb_total = 0.0
            eikonal_total = 0.0
            for entry in train_dataset:
                image = entry["image"].cuda().unsqueeze(0)
                camera = entry["camera"].cuda()
                xy = get_random_pixels_from_image(cfg.training.batch_size, cfg.data.image_size, camera)
                rays = get_rays_from_pixels(xy, cfg.data.image_size, camera)
                target = sample_images_at_xy(image, xy)
                prediction = model(rays)["feature"]
                rgb_loss = torch.mean((prediction - target) ** 2)
                eikonal = get_eikonal_loss(
                    model.implicit_fn,
                    cfg.training.batch_size,
                    cfg.training.bounds,
                    target.device,
                )
                loss = rgb_loss + params["eikonal_weight"] * eikonal
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                rgb_total += rgb_loss.item()
                eikonal_total += eikonal.item()
            final_rgb = rgb_total / len(train_dataset)
            final_eikonal = eikonal_total / len(train_dataset)
            if epoch % 5 == 0 or epoch == EPOCHS - 1:
                print(
                    f"ABLATION_PROGRESS {name} epoch={epoch:02d} "
                    f"rgb={final_rgb:.6f} eikonal={final_eikonal:.6f}",
                    flush=True,
                )

        model.eval()
        result = {
            **params,
            "epochs": EPOCHS,
            "train_rgb_mse": final_rgb,
            "train_eikonal": final_eikonal,
            **validation_metrics(model, val_dataset, cfg.data.image_size),
            **mesh_metrics(model.implicit_fn),
        }
        all_results[name] = result
        torch.save({"model": model.state_dict(), "result": result}, RESULT_DIR / f"{name}.pt")
        (RESULT_DIR / "results.json").write_text(json.dumps(all_results, indent=2))
        print(f"ABLATION_DONE {name} {json.dumps(result, sort_keys=True)}", flush=True)
        del model, optimizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
