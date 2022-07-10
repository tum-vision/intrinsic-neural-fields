"""
Adapted from
https://github.com/fbxiang/NeuTex/blob/master/models/nerf_atlas_radiance_model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .base_model import BaseModel
# from .decoder import GeometryMlpDecoder
# from .diff_ray_marching import find_ray_generation_method, ray_march
# from .diff_render_func import (
#     alpha_blend,
#     simple_tone_map,
#     radiance_render,
# )
from .atlasnet_networks import Atlasnet  # , chamfer_distance
from .atlasnet_inverse import InverseAtlasnet
from .texture_mlp import TextureMlpMix
# from .embedding import LpEmbedding
# import numpy as np
from copy import deepcopy
import os


DEFAULT_OPT = {
    "primitive_count": 1,  # Should be 1 when we use "sphere" as primitive type
    "primitive_type": "sphere",  # Seemed like that this works better than "square"
    "texture_num_layers": 6,  # The view-dependent model had 4 (block1), 2 (block2) as default
    "texture_decoder_width": 128,  # Default width
    "atlasnet_activation": "relu",
    "num_freqs": 10,
}


def make_neutex_for_pretraining():
    return NeuTex(DEFAULT_OPT)


def make_neutex_train_wrapper_default(model_config):
    pretrained_path = model_config["pretrained_path"]
    assert os.path.exists(pretrained_path)
    opt = deepcopy(DEFAULT_OPT)
    opt["num_freqs"] = model_config.get("k", 10)
    opt["use_logspace"] = model_config.get("use_logspace", True)
    opt["max_freq"] = model_config.get("max_freq")
    return NeuTexTrainWrapper(opt, pretrained_path=pretrained_path)


class NeuTexTrainWrapper(nn.Module):
    def __init__(self, opt, pretrained_path=None):
        super().__init__()
        self.opt = opt
        self.neutex = NeuTex(opt, pretrained_path=pretrained_path)

        # TODO(daniel): Set some fixed values here
        self.opt["loss_color_weight"] = 1  # Set in dtu.sh
        self.opt["loss_chamfer_weight"] = -1
        self.opt["loss_origin_weight"] = 1  # Set per default in nerf_atlas_radiance_model.py
        self.opt["loss_inverse_uv_weight"] = -1
        self.opt["loss_inverse_selection_weight"] = -1
        self.opt["loss_inverse_mapping_weight"] = 1  # Set in dtu.sh
        self.opt["loss_density_weight"] = -1

    def forward(self, batch, compute_inverse_mapping=True):
        output = self.neutex(batch, compute_inverse_mapping=compute_inverse_mapping)
        if self.training:
            return self.compute_loss(output, batch["expected_rgbs"]), output["rgbs"]
        return output["rgbs"]

    def compute_loss(self, output, gt_rgbs):
        loss_total = 0

        if self.opt.get("loss_color_weight", -1) > 0:
            loss_color = F.mse_loss(output["rgbs"], gt_rgbs)
            loss_total += self.opt.get("loss_color_weight") * loss_color

        """
        if self.opt.loss_bg_weight > 0:
            print("MASK LOSS")
            if "transmittance" in self.input:
                self.loss_bg = F.mse_loss(
                    self.output["transmittance"], self.input["transmittance"]
                )
            else:
                self.loss_bg = 0
            self.loss_total += self.opt.loss_bg_weight * self.loss_bg
        """

        if self.opt.get("loss_chamfer_weight", -1) > 0:
            raise NotImplementedError("Chamfer Loss not implemented!")

        # We cant use this
        if self.opt.get("loss_origin_weight", -1) > 0 and False:
            loss_origin = (
                ((output["points"] ** 2).sum(-2) - 1).clamp(min=0).sum()
            )
            loss_total += self.opt.get("loss_origin_weight") * loss_origin

        if self.opt.get("loss_inverse_uv_weight", -1) > 0:
            shape = output["points_2d_inverse"].shape
            inverse_points_2d = (
                output["points_2d_inverse"]
                    .view(shape[0], shape[2], -1, shape[2], shape[3])
                    .diagonal(dim1=1, dim2=3)
            ).permute(0, 1, 3, 2)

            loss_inverse_uv = F.mse_loss(
                inverse_points_2d,
                output["gt_points_2d"].expand_as(inverse_points_2d),
            )
            loss_total += self.opt.get("loss_inverse_uv_weight") * loss_inverse_uv

        if self.opt.get("loss_inverse_selection_weight", -1) > 0:
            weights_inverse_logits = output["weights_inverse_logits"]
            gt_weights = output["gt_primitive"].expand(
                weights_inverse_logits.shape[:-1]
            )
            loss_inverse_selection = F.cross_entropy(
                weights_inverse_logits.permute(0, 2, 1), gt_weights
            )
            loss_total += (
                    self.opt.get("loss_inverse_selection_weight") * loss_inverse_selection
            )

        if self.opt.get("loss_inverse_mapping_weight", -1) > 0:
            gt_points = output["points_original"]
            points = output["points_inverse"]
            ppw = output["points_inverse_primitive_weights"]
            # Note: we dont have this weight
            # pw = output["points_inverse_weights"]

            dist = ((gt_points[..., None, :] - points) ** 2).sum(-1)
            dist = (dist * ppw).sum(-1)
            # dist = (dist * pw).sum(-1)
            dist = dist.mean()

            loss_inverse_mapping = dist
            loss_total += (
                    self.opt.get("loss_inverse_mapping_weight") * loss_inverse_mapping
            )

        if self.opt.get("loss_density_weight", -1) > 0:
            loss_density = (
                (1 - output["points_density"] / 20).clamp(min=0).mean()
            )
            loss_total += self.opt.get("loss_density_weight") * loss_density

        return loss_total


"""
Some notes:

- Since we use primitive_type = 'sphere', the primitive count/patch count is always 1 according to the AtlasNet paper.
  The reason why we want to use multiple patches for primitive_type = 'square' is that it is not closed unlike the sphere.
- 
"""
# class NerfAtlasNetwork(nn.Module):
class NeuTex(nn.Module):
    def __init__(self, opt, pretrained_path=None):
        super().__init__()
        self.opt = opt

        # Getting rid off volume density
        """
        self.net_geometry_embedding = LpEmbedding(1, self.opt.geometry_embedding_dim)

        self.net_geometry_decoder = GeometryMlpDecoder(
            code_dim=0,
            pos_freqs=10,
            uv_dim=0,
            uv_count=0,
            brdf_dim=3,
            hidden_size=256,
            num_layers=10,
            requested_features={
                "density",
                # , "brdf"
            },
        )
        """

        primitive_type = self.opt.get("primitive_type", "sphere")
        assert primitive_type == "sphere"
        primitive_count = self.opt.get("primitive_count", 1)
        assert primitive_count == 1
        points_per_primitive = self.opt.get("points_per_primitive", 25000)

        # uv to 3D mapping
        self.net_atlasnet_uv_to_3d = Atlasnet(
            points_per_primitive,
            primitive_count,
            self.opt.get("atlasnet_activation"),
            primitive_type,
        )

        # 3D to uv mapping
        self.net_inverse_atlasnet_3d_to_uv = InverseAtlasnet(
            primitive_count,
            primitive_type=primitive_type,
        )

        # other types are not part of the release
        # assert opt.texture_decoder_type == "texture_view_mlp_mix"

        # Texture prediction from uv-map
        """
        self.net_texture = TextureViewMlpMix(
            opt.primitive_count,
            3,
            10,
            6,
            uv_dim=2 if self.opt.primitive_type == "square" else 3,
            layers=[int(x) for x in self.opt.texture_decoder_depth.split(",")],
            width=self.opt.texture_decoder_width,
            clamp=False,
        )
        """
        self.net_texture = TextureMlpMix(
            primitive_count=primitive_count,
            out_channels=3,  # RGB output
            num_freqs=self.opt.get("num_freqs"),
            uv_dim=(2 if primitive_type == "square" else 3),
            num_layers=self.opt.get("texture_num_layers"),
            width=self.opt.get("texture_decoder_width"),
            use_logspace=self.opt.get("use_logspace"),
            max_freq=self.opt.get("max_freq")
        )

        if pretrained_path is not None:
            self._init_after_mapping_pretrain(weights_path=pretrained_path)

        # We use ray-mesh intersection instead
        # self.raygen = find_ray_generation_method("cube")

    def _init_after_mapping_pretrain(self, weights_path):
        state = torch.load(weights_path)
        self.net_inverse_atlasnet_3d_to_uv.load_state_dict(state["net_inverse_atlasnet_3d_to_uv"])
        self.net_atlasnet_uv_to_3d.load_state_dict(state["net_atlasnet_uv_to_3d"])

    def forward(self, batch, compute_inverse_mapping=True, init_pass=False):
        output = {}

        # Initialization: 2D-to-3D-to-2D
        if init_pass:
            # Sample random uv-points and map them to 3D
            output["sampled_uv_points"], output["points3d"] = self.net_atlasnet_uv_to_3d.sample_uvs_and_transform_to_3d()

            output["sampled_uv_points"] = output["sampled_uv_points"].squeeze()  # num_points x uv_dim
            output["points3d"] = output["points3d"].squeeze()  # num_points x 3

            # Then, reverse. Transform the 3D points back to uv-map points
            output["inverted_uv_points"], _, _ = self.net_inverse_atlasnet_3d_to_uv(output["points3d"])
            output["inverted_uv_points"] = output["inverted_uv_points"].squeeze()  # num

            return output

        assert "xyz" in batch
        points_3d = batch["xyz"]  # N x 3

        output["points_original"] = deepcopy(points_3d)  # deepcopy, just to be sure...

        # Forward mapping from 3D to uv
        uv_points, weights, _ = self.net_inverse_atlasnet_3d_to_uv(points_3d)

        # Predict the RGB color
        output["rgbs"] = self.net_texture(uv_points, weights)

        # Inverse mapping from uv to 3D
        if compute_inverse_mapping:
            output["points_inverse"] = self.net_atlasnet_uv_to_3d(uv_points)
            output["points_inverse_primitive_weights"] = weights
            # We dont have the points_inverse_weights
            # output["points_inverse_weights"] = blend_weight

        return output

    """
    def forward(
        self,
        camera_position=None,
        ray_direction=None,
        background_color=None,
        compute_atlasnet=True,
        compute_inverse_mapping=False,
        compute_atlasnet_density=False,
    ):
        output = {}

        # The geometry embedding is a latent vector storing the 3D structure of the geometry
        zeros = torch.zeros(
            camera_position.shape[0], dtype=torch.long, device=camera_position.device
        )
        geometry_embedding = self.net_geometry_embedding(zeros)

        orig_ray_pos, ray_dist, ray_valid, ray_ts = self.raygen(
            camera_position, ray_direction, self.opt.sample_num, jitter=0.05
        )
        ray_pos = orig_ray_pos  # (N, rays, samples, 3)

        mlp_output = self.net_geometry_decoder(None, ray_pos)

        if compute_atlasnet:
            point_array_2d, points_3d = self.net_atlasnet(geometry_embedding)

            output["points"] = points_3d.view(
                points_3d.shape[0], -1, points_3d.shape[-1]
            ).permute(
                0, 2, 1
            )  # (N, 3, total_points)

            (
                output["points_2d_inverse"],
                output["weights_inverse"],
                output["weights_inverse_logits"],
            ) = self.net_inverse_atlasnet(
                geometry_embedding,
                points_3d.view(points_3d.shape[0], -1, points_3d.shape[-1]),
            )
            output["gt_primitive"] = self.net_atlasnet.get_label(points_3d.device)[None]
            output["gt_points_2d"] = torch.stack(point_array_2d, dim=1)[None]

            if compute_atlasnet_density:
                for param in self.net_geometry_decoder.parameters():
                    param.requires_grad_(False)
                output["points_density"] = self.net_geometry_decoder(
                    None, points_3d.view(points_3d.shape[0], -1, 1, 3)
                )["density"]
                for param in self.net_geometry_decoder.parameters():
                    param.requires_grad_(True)

        uv, weights, logits = self.net_inverse_atlasnet(geometry_embedding, ray_pos)

        point_features = self.net_texture(
            None, uv, ray_direction[:, :, None, :], weights
        )
        # point_features = torch.ones_like(point_features, device=point_features.device)

        points_inverse = None

        density = mlp_output["density"][..., None]
        radiance = point_features[..., :3]
        bsdf = [density, radiance]
        bsdf = torch.cat(bsdf, dim=-1)

        (
            ray_color,
            point_color,
            opacity,
            acc_transmission,
            blend_weight,
            background_transmission,
            background_blend_weight,
        ) = ray_march(
            ray_direction,
            ray_pos,
            ray_dist,
            ray_valid,
            bsdf,
            None,
            None,
            radiance_render,
            alpha_blend,
        )
        if background_color is not None:
            ray_color += (
                background_color[:, None, :] * background_blend_weight[:, :, None]
            )
        ray_color = simple_tone_map(ray_color)
        output["color"] = ray_color
        output["transmittance"] = background_blend_weight

        if compute_inverse_mapping:
            output["points_original"] = ray_pos
            if points_inverse is None:
                points_inverse = self.net_atlasnet.map(geometry_embedding, uv)
            output["points_inverse"] = points_inverse
            output["points_inverse_primitive_weights"] = weights
            output["points_inverse_weights"] = blend_weight

        return output
    """


"""
class NerfAtlasRadianceModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument(
            "--sample_num", required=True, type=int, help="number of samples along ray",
        )

        parser.add_argument(
            "--geometry_embedding_dim",
            required=True,
            type=int,
            help="the dimension for geometry embedding",
        )

        parser.add_argument(
            "--texture_decoder_type",
            required=True,
            type=str,
            choices=["texture_view_mlp_mix", "texture_view_mlp_mix_2"],
            help="type of texture decoder",
        )

        parser.add_argument(
            "--texture_decoder_width", type=int, default=128, help="network width"
        )

        parser.add_argument(
            "--texture_decoder_depth", type=str, default="4,2", help="network depth"
        )

        parser.add_argument(
            "--atlasnet_activation",
            required=True,
            type=str,
            choices=["relu", "softplus"],
            help="type activation",
        )

        parser.add_argument(
            "--loss_color_weight", required=True, type=float, help="rendering loss",
        )

        parser.add_argument(
            "--loss_bg_weight", required=True, type=float, help="transmittance loss",
        )

        parser.add_argument(
            "--loss_chamfer_weight",
            required=True,
            type=float,
            help="atlasnet chamfer loss",
        )

        parser.add_argument(
            "--loss_density_weight",
            required=True,
            type=float,
            help="penalize 2d points mapping to low-density region",
        )

        parser.add_argument(
            "--loss_origin_weight",
            required=False,
            default=1,
            type=float,
            help="penalize points too far away from origin",
        )

        parser.add_argument(
            "--loss_inverse_mapping_weight",
            required=True,
            type=float,
            help="inverse mapping loss",
        )

        parser.add_argument(
            "--loss_inverse_uv_weight",
            type=float,
            required=True,
            help="L2 loss on 3D to 2D UV",
        )

        parser.add_argument(
            "--loss_inverse_selection_weight",
            type=float,
            required=True,
            help="CE loss for uv map selection",
        )

        parser.add_argument(
            "--primitive_type",
            type=str,
            choices=["square", "sphere"],
            required=True,
            help="square or sphere",
        )

        parser.add_argument(
            "--primitive_count", type=int, required=True, help="number of atlas",
        )

        parser.add_argument(
            "--points_per_primitive",
            type=int,
            required=True,
            help="number of points per primitive",
        )

        parser.add_argument(
            "--sphere_init", type=int, default=-1, required=False, help="",
        )

    def initialize(self, opt):
        super().initialize(opt)
        self.model_names = ["nerf_atlas"]
        self.net_nerf_atlas = NerfAtlasNetwork(opt, self.device)

        assert self.opt.gpu_ids, "gpu is required"
        if self.opt.gpu_ids:
            self.net_nerf_atlas.to(self.device)
            self.net_nerf_atlas = torch.nn.DataParallel(
                self.net_nerf_atlas, self.opt.gpu_ids
            )

        if self.is_train:
            self.schedulers = []
            self.optimizers = []

            params = list(self.net_nerf_atlas.parameters())
            self.optimizer = torch.optim.Adam(params, lr=opt.lr)
            self.optimizers.append(self.optimizer)

        if self.opt.sphere_init > 0:
            import trimesh

            self.sphere = (
                torch.tensor(np.array(trimesh.creation.icosphere(4).vertices))
                .float()
                .to(self.device)
            )

    def set_input(self, input):
        self.input = {}
        for key in input:
            self.input[key] = input[key].to(self.device)

    def forward(self):
        use_atlasnet = (
            self.opt.loss_chamfer_weight > 0
            or self.opt.loss_origin_weight > 0
            or self.opt.loss_inverse_uv_weight > 0
            or self.opt.loss_inverse_selection_weight > 0
            or self.opt.loss_density_weight > 0
        )
        use_inverse_mapping = self.opt.loss_inverse_mapping_weight > 0
        use_atlasnet_density = self.opt.loss_density_weight > 0
        self.output = self.net_nerf_atlas(
            self.input["campos"],
            self.input["raydir"],
            self.input["background_color"],
            compute_atlasnet=use_atlasnet,
            compute_inverse_mapping=use_inverse_mapping,
            compute_atlasnet_density=use_atlasnet_density,
        )

        if "gt_image" in self.input:
            self.compute_loss()

        with torch.no_grad():
            if "gt_image" in self.input:
                self.visual_names = ["gt_image"]
                self.gt_image = self.input["gt_image"]
            if "color" in self.output:
                self.visual_names.append("ray_color")
                self.ray_color = self.output["color"]
            if "foreground_blend_weight" in self.output:
                self.visual_names.append("transmittance")
                self.transmittance = self.output["foreground_blend_weight"][
                    ..., None
                ].expand_as(self.ray_color)

    def compute_loss(self):
        self.loss_names = []
        self.loss_total = 0
        self.loss_names.append("total")

        if self.opt.loss_color_weight > 0:
            self.loss_color = F.mse_loss(self.output["color"], self.input["gt_image"])
            self.loss_total += self.opt.loss_color_weight * self.loss_color
            self.loss_names.append("color")

        if self.opt.loss_bg_weight > 0:
            if "transmittance" in self.input:
                self.loss_bg = F.mse_loss(
                    self.output["transmittance"], self.input["transmittance"]
                )
            else:
                self.loss_bg = 0
            self.loss_total += self.opt.loss_bg_weight * self.loss_bg
            self.loss_names.append("bg")

        if self.opt.loss_chamfer_weight > 0:
            if self.opt.sphere_init > 0:
                dist1, dist2 = chamfer_distance(
                    self.output["points"], self.sphere[None].permute(0, 2, 1)
                )
            else:
                dist1, dist2 = chamfer_distance(
                    self.output["points"], self.input["point_cloud"].permute(0, 2, 1)
                )
            self.loss_chamfer = torch.mean(dist1) + torch.mean(dist2)
            self.loss_total += self.opt.loss_chamfer_weight * self.loss_chamfer
            self.loss_names.append("chamfer")

        if self.opt.loss_origin_weight > 0:
            self.loss_origin = (
                ((self.output["points"] ** 2).sum(-2) - 1).clamp(min=0).sum()
            )
            self.loss_total += self.opt.loss_origin_weight * self.loss_origin
            self.loss_names.append("origin")

        if self.opt.loss_inverse_uv_weight > 0:
            shape = self.output["points_2d_inverse"].shape
            inverse_points_2d = (
                self.output["points_2d_inverse"]
                .view(shape[0], shape[2], -1, shape[2], shape[3])
                .diagonal(dim1=1, dim2=3)
            ).permute(0, 1, 3, 2)

            self.loss_inverse_uv = F.mse_loss(
                inverse_points_2d,
                self.output["gt_points_2d"].expand_as(inverse_points_2d),
            )
            self.loss_total += self.opt.loss_inverse_uv_weight * self.loss_inverse_uv
            self.loss_names.append("inverse_uv")

        if self.opt.loss_inverse_selection_weight > 0:
            weights_inverse_logits = self.output["weights_inverse_logits"]
            gt_weights = self.output["gt_primitive"].expand(
                weights_inverse_logits.shape[:-1]
            )
            self.loss_inverse_selection = F.cross_entropy(
                weights_inverse_logits.permute(0, 2, 1), gt_weights
            )
            self.loss_total += (
                self.opt.loss_inverse_selection_weight * self.loss_inverse_selection
            )
            self.loss_names.append("inverse_selection")

        if self.opt.loss_inverse_mapping_weight > 0:
            gt_points = self.output["points_original"]
            points = self.output["points_inverse"]
            ppw = self.output["points_inverse_primitive_weights"]
            pw = self.output["points_inverse_weights"]

            dist = ((gt_points[..., None, :] - points) ** 2).sum(-1)
            dist = (dist * ppw).sum(-1)
            dist = (dist * pw).sum(-1)
            dist = dist.mean()

            self.loss_inverse_mapping = dist
            self.loss_total += (
                self.opt.loss_inverse_mapping_weight * self.loss_inverse_mapping
            )
            self.loss_names.append("inverse_mapping")

        if self.opt.loss_density_weight > 0:
            self.loss_density = (
                (1 - self.output["points_density"] / 20).clamp(min=0).mean()
            )
            self.loss_total += self.opt.loss_density_weight * self.loss_density
            self.loss_names.append("density")

    def backward(self):
        self.optimizer.zero_grad()
        if self.opt.is_train:
            self.loss_total.backward()
            self.optimizer.step()

    def optimize_parameters(self):
        self.forward()
        self.backward()

    def test(self):
        with torch.no_grad():
            self.forward()

    def get_subnetworks(self):
        return {
            "nerf": self.net_nerf_atlas.module.net_geometry_decoder,
            "atlas": self.net_nerf_atlas.module.net_atlasnet,
            "inverse_atlas": self.net_nerf_atlas.module.net_inverse_atlasnet,
            "texture": self.net_nerf_atlas.module.net_texture,
        }

    def visualize_volume(self, res, block_size=64):
        with torch.no_grad():
            zeros = torch.zeros(1, dtype=torch.long, device=self.device)
            geometry_embedding = self.net_nerf_atlas.module.net_geometry_embedding(
                zeros
            )
            grid = np.stack(
                np.meshgrid(*([np.linspace(-1, 1, res)] * 3), indexing="ij"), axis=-1,
            )
            grid = grid.reshape(-1, 3)
            all_raypos = torch.FloatTensor(grid).to(self.device).view(1, -1, 1, 3)
            from tqdm import trange

            patch_size = block_size * block_size * block_size

            result = np.zeros((res ** 3, 4))

            for start in trange(0, all_raypos.shape[1], patch_size):
                raypos = all_raypos[:, start : start + patch_size]
                density = self.net_nerf_atlas.module.net_geometry_decoder(None, raypos)[
                    "density"
                ]
                uv, weights, logits = self.net_nerf_atlas.module.net_inverse_atlasnet(
                    geometry_embedding, raypos
                )

                output = (
                    torch.cat([density[..., None, None], uv], dim=-1)
                    .cpu()
                    .numpy()[0, :, 0, 0]
                )
                result[start : start + patch_size, : output.shape[-1]] = output

        return grid, result

    def visualize_atlas(self):
        zeros = torch.zeros(1, dtype=torch.long, device=self.device)
        geometry_embedding = self.net_nerf_atlas.module.net_geometry_embedding(zeros)
        point_array_2d, points_3d = self.net_nerf_atlas.module.net_atlasnet(
            geometry_embedding, 40000
        )
        uvs = torch.stack(point_array_2d, dim=-2)
        _, normals = self.net_nerf_atlas.module.net_atlasnet.map_and_normal(
            geometry_embedding, uvs
        )
        normals.permute(1, 0, 2)
        return points_3d[0], normals

    def visualize_texture_3d(self, resolution=512):
        with torch.no_grad():
            grid = np.stack(
                np.meshgrid(
                    np.linspace(-1.0, 1.0, resolution),
                    np.linspace(-1.0, 1.0, resolution),
                    indexing="ij",
                ),
                axis=-1,
            )
            grid = np.stack([grid] * self.opt.primitive_count, axis=-2)
            grid = torch.from_numpy(grid)[None].float().to(self.device)

            import trimesh

            mesh = trimesh.creation.icosphere(6)
            grid = torch.tensor(mesh.vertices).to(self.device).float()
            grid = grid.view(1, grid.shape[0], 1, 1, 3)

            textures = []
            for tex in range(self.opt.primitive_count):
                weights = (
                    torch.zeros(grid.shape[:3] + (self.opt.primitive_count,))
                    .float()
                    .to(self.device)
                )
                weights[0, :, :, tex] = 1
                textures.append(
                    self.net_nerf_atlas.module.net_texture(None, grid, weights)[0]
                )
            return grid.squeeze(), [t.squeeze() for t in textures]

    def visualize_mesh_3d(
        self, resolution=512, viewdir=[0, 0, 1], icosphere_division=6
    ):
        if self.opt.texture_decoder_type == "texture_view_mlp_mix":
            clamp_texture = True
        elif self.opt.texture_decoder_type == "texture_view_mlp_mix_2":
            clamp_texture = False

        with torch.no_grad():
            import trimesh

            mesh = trimesh.creation.icosphere(icosphere_division)
            grid = torch.tensor(mesh.vertices).to(self.device).float()
            grid = grid.view(1, grid.shape[0], 1, 1, 3)

            zeros = torch.zeros(1, dtype=torch.long, device=self.device)
            geometry_embedding = self.net_nerf_atlas.module.net_geometry_embedding(
                zeros
            )
            vertices = self.net_nerf_atlas.module.net_atlasnet.map(
                geometry_embedding, grid
            )

            meshes = []
            for v in vertices.unbind(-2):
                mesh = trimesh.creation.icosphere(icosphere_division)
                mesh.vertices = v.squeeze().data.cpu().numpy()
                meshes.append(mesh)

            textures = []
            for tex in range(self.opt.primitive_count):
                weights = (
                    torch.zeros(grid.shape[:3] + (self.opt.primitive_count,))
                    .float()
                    .to(self.device)
                )
                weights[0, :, :, tex] = 1

                viewdir = (
                    torch.tensor(viewdir)
                    .float()
                    .to(grid.device)
                    .expand(grid.shape[:-2] + (3,))
                )

                textures.append(
                    self.net_nerf_atlas.module.net_texture(
                        None, grid, viewdir, weights
                    )[0]
                )

            return meshes, [t.squeeze() for t in textures]
"""
