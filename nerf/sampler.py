import torch


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        z_vals = torch.linspace(
            self.min_depth,
            self.max_depth,
            self.n_pts_per_ray,
            device=ray_bundle.origins.device
        )

        sample_points = (
                ray_bundle.origins[:, None, :]
                + ray_bundle.directions[:, None, :] * z_vals[None, :, None]
        )

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals[None, :, None].expand(
                sample_points.shape[0], -1, -1
            ),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}
