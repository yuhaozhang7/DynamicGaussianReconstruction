from scene.mvsplat.model.encoder.backbone import BackboneMultiview

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

try:
    from typing import Optional
except ImportError:
    from typing_extensions import Optional

try:
    from typing import List
except ImportError:
    from typing_extensions import List


from dataclasses import dataclass
from einops import rearrange

import torch
from torch import nn
from scene.mvsplat.model.encoder.epipolar.epipolar_sampler import EpipolarSampler
from scene.mvsplat.model.encodings.positional_encoding import PositionalEncoding
from scene.mvsplat.model.encoder.common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from scene.mvsplat.model.encoder.costvolume.depth_predictor_multiview import DepthPredictorMultiView
from scene.mvsplat.geometry.projection import sample_image_grid
import numpy as np
from jaxtyping import Float

from collections import OrderedDict
import os
config = {
    'num_context_views' : 2,
    'mean_feat' : False,
    'd_feature' : 128,
    'downscale_factor' : 4,
    'wo_backbone_cross_attn':False,
    'use_epipolar_trans' : False,
    'unimatch_weights_path' : None,
    'num_depth_candidates' : 128,
    'costvolume_unet_feat_dim':128,
    'costvolume_unet_channel_mult':[1,1,1],
    'costvolume_unet_attn_res': [4],
    'num_surfaces' : 1,
    'gaussians_per_pixel' : 1,
    'depth_unet_feat_dim' : 32,
    'depth_unet_attn_res' : [16],
    'depth_unet_channel_mult' : [1,1,1,1,1],
    'wo_depth_refine':False,
    'wo_cost_volume':False,
    'wo_cost_volume_refine':False,
    'multiview_trans_attn_split':2,
    'gaussian_scale_max' : 15.0,
    'gaussian_scale_min' : 0.5,
    'sh_degree' : 4,

    'mode' : 'train'


}
# config['unimatch_weights_path'] = 'checkpoints/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth'




class OpacityMappingCfg:
    def __init__(self,initial,final,warm_up):
        self.initial = initial
        self.final = final
        self.warm_up = warm_up



class MVSplat(torch.nn.Module):
    def __init__(self,config,device,use_depth=False,use_cnn_feats=False,with_pe=True,feature_map_ch_dim=128):
        super(MVSplat, self).__init__()
        self.config = config
        self.use_depth = use_depth
        self.use_cnn_feats = use_cnn_feats
        if config['use_epipolar_trans']:
            self.epipolar_sampler = EpipolarSampler(
                num_views=config['num_context_views'],
                num_samples=32,
            )
            pe = PositionalEncoding(10)
            self.depth_encoding = nn.Sequential(
                pe,
                nn.Linear(pe.d_out(1), config['d_feature']),
            )
        self.backbone = BackboneMultiview(
            feature_channels=config['d_feature'],
            downscale_factor=config['downscale_factor'],
            no_cross_attn=config['wo_backbone_cross_attn'],
            use_epipolar_trans=config['use_epipolar_trans'],
        ).to(device)
        ckpt_path = config['unimatch_weights_path']

        if config['mode'] == 'train':
            if config['unimatch_weights_path'] is None:
                print("==> Init multi-view transformer backbone from scratch")
            else:
                print("==> Load multi-view transformer backbone checkpoint: %s" % ckpt_path)
                unimatch_pretrained_model = torch.load(ckpt_path)["model"]
                updated_state_dict = OrderedDict(
                    {
                        k: v
                        for k, v in unimatch_pretrained_model.items()
                        if k in self.backbone.state_dict()
                    }
                )
                # NOTE: when wo cross attn, we added ffns into self-attn, but they have no pretrained weight
                is_strict_loading = not config['wo_backbone_cross_attn']
                self.backbone.load_state_dict(updated_state_dict, strict=is_strict_loading)


        if self.use_depth:
            gaussian_adapter_cfg = GaussianAdapterCfg(gaussian_scale_max=15.0,gaussian_scale_min=0.5,sh_degree=config['sh_degree']) # TODO CHANGE THESE VALUES MAYBE
            self.gaussian_adapter = GaussianAdapter(gaussian_adapter_cfg).to(device)
            self.depth_predictor = DepthPredictorMultiView(
                feature_channels=config['d_feature'],
                upscale_factor=config['downscale_factor'],
                num_depth_candidates=config['num_depth_candidates'],
                costvolume_unet_feat_dim=config['costvolume_unet_feat_dim'],
                costvolume_unet_channel_mult=tuple(config['costvolume_unet_channel_mult']),
                costvolume_unet_attn_res=tuple(config['costvolume_unet_attn_res']),
                gaussian_raw_channels=config['num_surfaces'] * (self.gaussian_adapter.d_in + 2),
                gaussians_per_pixel=config['gaussians_per_pixel'],
                num_views=config['num_context_views'],
                depth_unet_feat_dim=config['depth_unet_feat_dim'],
                depth_unet_attn_res=config['depth_unet_attn_res'],
                depth_unet_channel_mult=config['depth_unet_channel_mult'],
                wo_depth_refine=config['wo_depth_refine'],
                wo_cost_volume=config['wo_cost_volume'],
                wo_cost_volume_refine=config['wo_cost_volume_refine'],
            ).to(device)
            self.opacity_mapping_cfg = OpacityMappingCfg(0.0,0.0,1.0) # TODO CHANGE THESE VALUES MAYBE
        else:
            self.mlp = MLP(out_channels=feature_map_ch_dim,with_pe=with_pe)

    def map_pdf_to_opacity(
        self,
        pdf: Float[torch.Tensor, " *batch"],
        global_step: int,
    ) -> Float[torch.Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.opacity_mapping_cfg
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))
    def camera_to_context(self,cameras,num_views,resize=None):

        device = cameras[0].original_image.device
        context = {}
        extrinsics = np.zeros((1,num_views,4,4))
        for i in range(len(cameras)):
            camera = cameras[i]
            extrinsics[0,i,:3,:3] = camera.R
            extrinsics[0,i,:3,3] = camera.T
            extrinsics[0,i,3,3] = 1
        context['extrinsics'] = torch.tensor(extrinsics,dtype=torch.float32).to(device=device)
        intrinsics = np.zeros((1, num_views, 3, 3))
        for i in range(len(cameras)):
            camera = cameras[i]
            intrinsics[0, i, 0, 0] = camera.FoVx
            intrinsics[0, i, 1, 1] = camera.FoVy
            intrinsics[0,i,2,2] = 1
            intrinsics[0, i, 0, 1] = 0 # TODO SKEW PROBABLY 0
            # PRINCIPAL POINTS ARE USUALLY THE CENTER (IDK COULDNT FIND IN CAMERA)
            intrinsics[0, i, 0, 2] = 0.5
            intrinsics[0, i, 1, 2] = 0.5
        context['intrinsics'] = torch.tensor(intrinsics,dtype=torch.float32).to(device=device)
        context['far'] = torch.zeros((1,num_views),dtype=torch.float32).to(device=device)
        context['near'] = torch.zeros((1, num_views), dtype=torch.float32).to(device=device)
        for i in range(len(cameras)):
            camera = cameras[i]
            context['far'][0,i] = camera.zfar
            context['near'][0,i] = camera.znear
        if resize:
            context['image'] = torch.zeros((1, num_views, cameras[0].original_image.shape[0],
                                            resize, resize),
                                           dtype=torch.float32).to(device=device)
        else:
            context['image'] = torch.zeros((1,num_views,cameras[0].original_image.shape[0],cameras[0].original_image.shape[1],cameras[0].original_image.shape[2]),dtype=torch.float32).to(device=device)
        for i in range(len(cameras)):
            camera = cameras[i]
            if resize:
                context['image'][0,i] = torch.nn.functional.interpolate(camera.original_image.unsqueeze(0), size=(resize, resize), mode='bilinear', align_corners=False).squeeze(0)
            else:
                context['image'][0, i] = camera.original_image
        return context


    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ):
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        # Encode the context images.
        if self.config['use_epipolar_trans']:
            epipolar_kwargs = {
                "epipolar_sampler": self.epipolar_sampler,
                "depth_encoding": self.depth_encoding,
                "extrinsics": context["extrinsics"],
                "intrinsics": context["intrinsics"],
                "near": context["near"],
                "far": context["far"],
            }
        else:
            epipolar_kwargs = None
        trans_features, cnn_features = self.backbone(
            context["image"],
            attn_splits=self.config['multiview_trans_attn_split'],
            return_cnn_features=True,
            epipolar_kwargs=epipolar_kwargs,
        )

        # Sample depths from the resulting features.
        if self.use_cnn_feats:
            in_feats = cnn_features
        else:
            in_feats = trans_features

        if self.use_depth:
            extra_info = {}
            extra_info['images'] = rearrange(context["image"], "b v c h w -> (v b) c h w")
            extra_info["scene_names"] = scene_names
            gpp = self.config['gaussians_per_pixel']
            depths, densities, raw_gaussians = self.depth_predictor(
                in_feats,
                context["intrinsics"],
                context["extrinsics"],
                context["near"],
                context["far"],
                gaussians_per_pixel=gpp,
                deterministic=deterministic,
                extra_info=extra_info,
                cnn_features=cnn_features,
            )

            # Convert the features and depths into Gaussians.
            xy_ray, _ = sample_image_grid((h, w), device)
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
            gaussians = rearrange(
                raw_gaussians,
                "... (srf c) -> ... srf c",
                srf=self.config['num_surfaces'],
            )
            offset_xy = gaussians[..., :2].sigmoid()
            pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
            xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
            gpp = self.config['gaussians_per_pixel']
            gaussians = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                self.map_pdf_to_opacity(densities, global_step) / gpp,
                rearrange(
                    gaussians[..., 2:],
                    "b v r srf c -> b v r srf () c",
                ),
                (h, w),
            )

            # Dump visualizations if needed.
            if visualization_dump is not None:
                visualization_dump["depth"] = rearrange(
                    depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
                )
                visualization_dump["scales"] = rearrange(
                    gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
                )
                visualization_dump["rotations"] = rearrange(
                    gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
                )

            # Optionally apply a per-pixel opacity.
            opacity_multiplier = 1

            return rearrange(
                    gaussians.means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ), rearrange(
                    gaussians.covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ), rearrange(
                    gaussians.harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ), rearrange(
                    opacity_multiplier * gaussians.opacities,
                    "b v r srf spp -> b (v r srf spp)",
                )
        else:
            if config['mean_feat']:
                feat = torch.mean(in_feats, dim=1)
            else:
                feat = in_feats[:, 0]
            return self.mlp(feat)

    def save_weights(self,path,iteration):
        if os.path.exists(path + '/model_saves'):
            torch.save(self.state_dict(),path + '/model_saves/mvsplat_save_'+str(iteration)+'.pth')
        else:
            os.mkdir(path + '/model_saves')
            torch.save(self.state_dict(), path + '/model_saves/mvsplat_save_' + str(iteration) + '.pth')
    def load_weights(self,path):
        self.load_state_dict(torch.load(path))




def rotation_matrix_to_quaternion(R):
    """
    Differentiable conversion of a 3x3 rotation matrix to a quaternion (w, x, y, z).
    Args:
        R: torch.Tensor of shape (3, 3) or (batch_size, 3, 3)
    Returns:
        quaternion: torch.Tensor of shape (4,) or (batch_size, 4)
    """
    batch_mode = R.ndim == 3
    if not batch_mode:
        R = R.unsqueeze(0)  # Add batch dimension

    # Trace of the matrix
    t = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Initialize quaternion tensor
    q = torch.zeros((R.shape[0], 4), device=R.device, dtype=R.dtype)

    # Case 1: t > 0
    positive_trace_mask = t > 0
    t_positive = t[positive_trace_mask]
    q[positive_trace_mask, 0] = torch.sqrt(1.0 + t_positive) / 2.0  # w
    q[positive_trace_mask, 1] = (R[positive_trace_mask, 2, 1] - R[positive_trace_mask, 1, 2]) / (4.0 * q[positive_trace_mask, 0])
    q[positive_trace_mask, 2] = (R[positive_trace_mask, 0, 2] - R[positive_trace_mask, 2, 0]) / (4.0 * q[positive_trace_mask, 0])
    q[positive_trace_mask, 3] = (R[positive_trace_mask, 1, 0] - R[positive_trace_mask, 0, 1]) / (4.0 * q[positive_trace_mask, 0])

    # Case 2: t <= 0
    negative_trace_mask = ~positive_trace_mask
    R_neg = R[negative_trace_mask]
    diag = torch.stack([R_neg[:, 0, 0], R_neg[:, 1, 1], R_neg[:, 2, 2]], dim=1)
    max_diag_index = torch.argmax(diag, dim=1)

    for i in range(3):
        submask = negative_trace_mask.clone()
        submask[negative_trace_mask] &= max_diag_index == i

        if i == 0:  # R[0, 0] is the largest diagonal element
            q[submask, 1] = torch.sqrt(1.0 + 2.0 * R[submask, 0, 0] - t[submask]) / 2.0  # x
            q[submask, 0] = (R[submask, 2, 1] - R[submask, 1, 2]) / (4.0 * q[submask, 1])  # w
            q[submask, 2] = (R[submask, 0, 1] + R[submask, 1, 0]) / (4.0 * q[submask, 1])  # y
            q[submask, 3] = (R[submask, 0, 2] + R[submask, 2, 0]) / (4.0 * q[submask, 1])  # z
        elif i == 1:  # R[1, 1] is the largest diagonal element
            q[submask, 2] = torch.sqrt(1.0 + 2.0 * R[submask, 1, 1] - t[submask]) / 2.0  # y
            q[submask, 0] = (R[submask, 0, 2] - R[submask, 2, 0]) / (4.0 * q[submask, 2])  # w
            q[submask, 1] = (R[submask, 0, 1] + R[submask, 1, 0]) / (4.0 * q[submask, 2])  # x
            q[submask, 3] = (R[submask, 1, 2] + R[submask, 2, 1]) / (4.0 * q[submask, 2])  # z
        elif i == 2:  # R[2, 2] is the largest diagonal element
            q[submask, 3] = torch.sqrt(1.0 + 2.0 * R[submask, 2, 2] - t[submask]) / 2.0  # z
            q[submask, 0] = (R[submask, 1, 0] - R[submask, 0, 1]) / (4.0 * q[submask, 3])  # w
            q[submask, 1] = (R[submask, 0, 2] + R[submask, 2, 0]) / (4.0 * q[submask, 3])  # x
            q[submask, 2] = (R[submask, 1, 2] + R[submask, 2, 1]) / (4.0 * q[submask, 3])  # y

    if not batch_mode:
        q = q.squeeze(0)  # Remove batch dimension if input was not batched

    return q


class MLP(torch.nn.Module):
    def __init__(self,out_channels,with_pe):
        super(MLP,self).__init__()

        self.feature_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        def post_decoder(in_channels, mid_channels, out_channels):
            return nn.Sequential(
                nn.Linear(in_channels, mid_channels),
                nn.LayerNorm(mid_channels),
                nn.ReLU(inplace=True),
                nn.Linear(mid_channels, mid_channels),
                nn.LayerNorm(mid_channels),
                nn.ReLU(inplace=True),
                nn.Linear(mid_channels, out_channels)
            )

        self.color_features_decoder1 = post_decoder(out_channels, 256, 16)
        self.color_features_decoder2 = post_decoder(out_channels, 256, 16)
        self.color_features_decoder3 = post_decoder(out_channels, 256, 16)
        self.mean_decoder = post_decoder(out_channels, 256, 3)
        self.opacity_decoder = post_decoder(out_channels, 256, 1)
        self.scale_decoder = post_decoder(out_channels, 256, 1)  # assume isometric scaling
        self.with_pe = with_pe

    def positional_encoding(self, h, w, out_channel):
        assert out_channel % 2 == 0, "out_channel must be even for sinusoidal encoding"
        pe = np.zeros((h * w, out_channel))
        position = np.arange(h * w)[:, np.newaxis]
        div_term = np.exp(np.arange(0, out_channel, 2) * -(np.log(10000.0) / out_channel))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe
    def forward(self,features):
        features = self.feature_conv(features)
        bs, out_channel, h, w = features.shape
        features = features.permute(2, 3, 0, 1)  # Now features.shape == (height, width, batch_size, out_channel)
        features = features.reshape(h * w, out_channel)
        if self.with_pe:
            pe = torch.tensor(self.positional_encoding(h, w, out_channel), dtype=features.dtype).to("cuda")
            features = features + pe
        mean = self.mean_decoder(features).reshape(bs, -1, 3)
        # mean = torch.tanh(mean) * 5

        color_features1 = self.color_features_decoder1(features)
        color_features2 = self.color_features_decoder2(features)
        color_features3 = self.color_features_decoder3(features)
        color_features = torch.stack([color_features1, color_features2, color_features3],
                                     dim=2)  # Shape: [h * w, 16, 3]
        color_features = color_features.unsqueeze(0)

        # color_features = self.color_features_decoder(features).reshape(bs, -1, 16, 3)
        # color_features = torch.tanh(color_features) * 10

        opacity = self.opacity_decoder(features).reshape(bs, -1, 1)  # no need for signmoid

        scale = self.scale_decoder(features).reshape(bs, -1, 1)  # no need for exponential
        # rotation = self.rotation_decoder

        return mean, color_features, opacity, scale, features





