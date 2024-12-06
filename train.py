#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, l2_loss, ssim, kl_divergence
from utils.dataset_utils import sample_small_dataset
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel, GaussianPredictor
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
from torchvision.transforms import functional as F
from PIL import Image
import random

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# os.environ['WANDB_MODE'] = 'disabled'

def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    feature_map_ch_dim = 128
    input_images_size = 20
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(dataset.is_blender, dataset.is_6dof, feature_map_ch_dim)
    deform.train_setting(opt)
    gaussian_predictor = GaussianPredictor(input_images_size, 3 * input_images_size, feature_map_ch_dim).cuda()
    gaussian_predictor_optimizer = torch.optim.Adam(gaussian_predictor.parameters(), lr=1e-4)

    scene = Scene(dataset, gaussians)
    sampled_dataset = sample_small_dataset(scene, num=80, size=20, diff=6, reverse=True)
    # gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    for iteration in range(1, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # Pick a random Camera
        if viewpoint_stack is None:
            cameras = random.choice(sampled_dataset)
            viewpoint_stack = cameras.copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        mean, color_features, opacity, scale, img_features = gaussian_predictor(cameras, None)
        gaussians._xyz = mean[0]
        gaussians._features_img = img_features[0]
        gaussians._features_dc = color_features[0, :, 0:1]
        gaussians._features_rest = color_features[0, :, 1:]
        gaussians._opacity = opacity[0]
        gaussians._scaling = torch.ones(scale.size(1), 3, device='cuda') * scale[0, :]
        gaussians._rotation = torch.zeros(scale.size(1), 4, device='cuda')
        gaussians._rotation[:, 0] = 1.0

        # log min & max of features, opacity and scaling
        wandb.log({"min_features": torch.min(gaussians._features_dc).item(),
                   "max_features": torch.max(gaussians._features_dc).item(),
                   "min_opacity": torch.min(gaussians._opacity).item(),
                   "max_opacity": torch.max(gaussians._opacity).item(),
                   "min_scaling": torch.min(gaussians._scaling).item(),
                   "max_scaling": torch.max(gaussians._scaling).item(),
                   "min_xyz": torch.min(gaussians._xyz).item(),
                   "max_xyz": torch.max(gaussians._xyz).item()
                   }, step=iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        opt.warm_up = 1000
        if iteration == opt.warm_up:
            print("Start training deformable network")
        
        loss = torch.tensor(0.0, requires_grad=True, device='cuda')

        for viewpoint_cam in viewpoint_stack:

            if dataset.load2gpu_on_the_fly:
                viewpoint_cam.load2device()
            fid = viewpoint_cam.fid

            if iteration < opt.warm_up:
                d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
            else:
                N = gaussians.get_xyz.shape[0]
                time_input = fid.unsqueeze(0).expand(N, -1)

                ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N,
                                                                                                -1) * time_interval * smooth_term(
                    iteration)
                
                if iteration < opt.warm_up + 5000:
                    d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise, gaussians.get_features_img.detach())
                else:
                    d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise, gaussians.get_features_img)

            # Render
            render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling,
                                dataset.is_6dof)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
                "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
            # depth = render_pkg_re["depth"]
            wandb.log({"min_image": torch.min(image).item(),
                    "max_image": torch.max(image).item()}, step=iteration)

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll2 = l2_loss(image, gt_image)
            loss = loss + Ll2

        loss /= len(viewpoint_stack)
        viewpoint_stack = None
        loss.backward()
        max_grad = max(param.grad.abs().max().item() for param in gaussian_predictor.parameters() if param.grad is not None)
        wandb.log({"max_grad": max_grad}, step=iteration)

        torch.nn.utils.clip_grad_norm_(gaussian_predictor.parameters(), max_norm=1.0)

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            if iteration % 1000 == 0:
                tensor = image * 255.0 
                tensor = tensor.to('cpu').byte()
                tensor = F.to_pil_image(tensor)
                os.makedirs(f"output/{PROJECT_NAME}/images", exist_ok=True)
                tensor.save(f"output/{PROJECT_NAME}/images/image_{iteration}.png")

            # Keep track of max radii in image-space for pruning
            # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
            #                                                      radii[visibility_filter])

            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll2, loss, l2_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof)
            if iteration in testing_iterations:
                wandb.log({"psnr": cur_psnr}, step=iteration)
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)
                gaussian_predictor.save_weights(args.model_path, iteration)

            wandb.log({"loss": loss.item(),
                       "ema_loss": ema_loss_for_log}, step=iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussian_predictor_optimizer.step()
                deform.optimizer.step()

                gaussian_predictor_optimizer.zero_grad()
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    features_img = scene.gaussians.get_features_img
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input, features_img.detach())
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)[
                            "render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    PROJECT_NAME = "dnerf_standup_small"
    wandb.init(
        project="mvp_project",
        name=PROJECT_NAME,
    )

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[1] + list(range(1000, 400_001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[1] +list(range(1000, 400_001, 1000)))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    setattr(args, "model_path", f"output/{PROJECT_NAME}")

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    # All done
    print("\nTraining complete.")

    wandb.finish()
