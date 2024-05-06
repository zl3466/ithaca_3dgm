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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import matplotlib.pyplot as plt
import numpy as np
import cv2

def normalize(loss):
    # Convert to numpy array
    loss_matrix = loss.cpu().detach().numpy()

    # Check if the image is grayscale (2D) or color (3D)
    if loss_matrix.ndim == 2:
        # For grayscale, normalization can be done directly
        normalized_loss = loss_matrix
    elif loss_matrix.ndim == 3:
        # Transpose and calculate the mean across channels for color images
        loss_matrix = np.transpose(loss_matrix, (1, 2, 0))
        loss_matrix = np.mean(loss_matrix, axis=-1)
        normalized_loss = loss_matrix
    else:
        raise ValueError("Input loss must be either 2D or 3D")

    # Normalize the loss matrix
    min_loss = np.min(normalized_loss)
    max_loss = np.max(normalized_loss)

    # Avoid division by zero in case all pixel values are the same
    if max_loss - min_loss != 0:
        normalized_loss = (normalized_loss - min_loss) / (max_loss - min_loss)
    else:
        # If the image is constant, the normalized result is zero
        normalized_loss = normalized_loss - min_loss

    return normalized_loss

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth_map_vis")
    depth_path2 = os.path.join(model_path, name, "ours_{}".format(iteration), "depth_map")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(depth_path2, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        rendered_depth = render_pkg["depth_map"]
        gt = view.original_image[0:3, :, :]

        rendered_depth_cpu = rendered_depth.cpu()  # 将张量移动到 CPU
        np.save(os.path.join(depth_path2, '{0:05d}'.format(idx) + ".npy"), rendered_depth_cpu.numpy())

        rendered_depth_normal = normalize(rendered_depth.squeeze(0))
        rendered_depth_magma = cv2.applyColorMap((rendered_depth_normal * 255).astype(np.uint8), cv2.COLORMAP_JET)
        

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        cv2.imwrite(os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"), rendered_depth_magma)
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)