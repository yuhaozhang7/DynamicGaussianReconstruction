import torch
import torch.nn as nn
import torch.nn.functional as F
from scene.u_net import UNet
# from scene.__init__ import Scene
import random
import numpy as np
import os
from utils.system_utils import searchForMaxIteration

class GaussianPredictor(nn.Module):

    def __init__(self, in_size, in_channels, out_channels):
        super(GaussianPredictor, self).__init__()

        self.in_size = in_size  # number of input images

        self.unet = UNet(in_channels, out_channels)

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

        # self.color_features_decoder = post_decoder(out_channels, 256, 48)
        self.color_features_decoder1 = post_decoder(out_channels, 256, 16)
        self.color_features_decoder2 = post_decoder(out_channels, 256, 16)
        self.color_features_decoder3 = post_decoder(out_channels, 256, 16)
        self.mean_decoder = post_decoder(out_channels, 256, 3)
        self.opacity_decoder = post_decoder(out_channels, 256, 1)
        self.scale_decoder = post_decoder(out_channels, 256, 1)  # assume isometric scaling


    def positional_encoding_images(self, selected_images, num_images, channels_per_image=3, height=128, width=128):

        d_model = height * width

        pe = torch.zeros(num_images, d_model, device='cuda')  # Shape (num_images, d_model)
        position = torch.arange(0, num_images, dtype=torch.float32, device=pe.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=pe.device).float() * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Sinusoidal encoding for even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Sinusoidal encoding for odd indices

        pe = pe.view(num_images, 1, height, width)  # Shape (num_images, 1, height, width)

        images_with_pe = []
        for i in range(num_images):
            image_channels = selected_images[i * channels_per_image:(i + 1) * channels_per_image]  # Extract image channels
            image_with_pe = torch.cat([image_channels, pe[i]], dim=0)  # Append PE as an extra channel
            # image_with_pe = image_channels + pe[i]
            images_with_pe.append(image_with_pe)

        return torch.cat(images_with_pe, dim=0) 


    def positional_encoding_features(self, h, w, out_channel):
        assert out_channel % 2 == 0, "out_channel must be even for sinusoidal encoding"
        pe = np.zeros((h * w, out_channel))
        position = np.arange(h * w)[:, np.newaxis]
        div_term = np.exp(np.arange(0, out_channel, 2) * -(np.log(10000.0) / out_channel))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe


    def forward(self, cameras, target_cam=None):

        cameras.sort(key=lambda cam: cam.fid.item())
        num_samples = min(self.in_size, len(cameras))
        indices = [round(i * (len(cameras) - 1) / (num_samples - 1)) for i in range(num_samples)]
        cameras = [cameras[i] for i in indices]
        # cameras.append(target_cam)

        selected_images = torch.cat([viewpoint_cam.original_image.cuda() for viewpoint_cam in cameras], dim=0)
        selected_images = selected_images.unsqueeze(0)

        features = self.unet(selected_images)
        features = self.feature_conv(features)
        bs, out_channel, h, w = features.shape
        features = features.permute(2, 3, 0, 1)  # Now features.shape == (height, width, batch_size, out_channel)
        features = features.reshape(h * w, out_channel)

        pe = torch.tensor(self.positional_encoding_features(h, w, out_channel), dtype=features.dtype).to("cuda")
        features = features + pe

        mean = self.mean_decoder(features).reshape(bs, -1, 3)

        color_features1 = self.color_features_decoder1(features)
        color_features2 = self.color_features_decoder2(features)
        color_features3 = self.color_features_decoder3(features)
        color_features = torch.stack([color_features1, color_features2, color_features3], dim=2)  # Shape: [h * w, 16, 3]
        color_features = color_features.unsqueeze(0)

        opacity = self.opacity_decoder(features).reshape(bs, -1, 1)  # no need for signmoid

        scale = self.scale_decoder(features).reshape(bs, -1, 1)  # no need for exponential

        return mean, color_features, opacity, scale, features.unsqueeze(0)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "gaussian_predictor/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(out_weights_path, 'gaussian_predictor.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "gaussian_predictor"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "gaussian_predictor/iteration_{}/gaussian_predictor.pth".format(loaded_iter))
        self.load_state_dict(torch.load(weights_path))
