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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, visualize_depth
from utils.graphics_utils import fov2focal
import copy
import torch
from torchvision.utils import save_image
import sys

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        #resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        resolution = round(orig_h/(resolution_scale * args.resolution)), round(orig_w/(resolution_scale * args.resolution))
        scale = resolution_scale * args.resolution
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        #resolution = (int(orig_w / scale), int(orig_h / scale), 3)
        resolution = (int(orig_h / scale), int(orig_w / scale), 3)

    if cam_info.cx:
        cx = cam_info.cx / scale
        cy = cam_info.cy / scale
        fy = cam_info.fy / scale
        fx = cam_info.fx / scale
    else:
        cx = None
        cy = None
        fy = None
        fx = None

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None
    
    dynamic_mask = copy.deepcopy(cam_info.mask)*255
    dynamic_mask = torch.round(PILtoTorch(dynamic_mask, resolution))
    # change it back to 0 and 1
    
    pts_depth = None
    if cam_info.pointcloud_camera is not None:
        h, w = gt_image.shape[1:]
        h = h*2
        w = w*2
        K = np.eye(3)
        if cam_info.cx:
            print(scale)
            K[0, 0] = fx / scale
            K[1, 1] = fy / scale
            K[0, 2] = cx / scale
            K[1, 2] = cy / scale
        else:
            K[0, 0] = fov2focal(cam_info.FovX, w)
            K[1, 1] = fov2focal(cam_info.FovY, h)
            K[0, 2] = cam_info.width / 2
            K[1, 2] = cam_info.height / 2
        pts_depth = np.zeros([1, h, w])
        point_camera = cam_info.pointcloud_camera
        uvz = point_camera[point_camera[:, 2] > 0]
        uvz = uvz @ K.T
        uvz[:, :2] /= (uvz[:, 2:]*5)
        uvz = uvz[uvz[:, 1] >= 0]
        uvz = uvz[uvz[:, 1] < h]
        uvz = uvz[uvz[:, 0] >= 0]
        uvz = uvz[uvz[:, 0] < w]
        uv = uvz[:, :2]
        uv = uv.astype(int)
        # TODO: may need to consider overlap
        pts_depth[0, uv[:, 1], uv[:, 0]] = uvz[:, 2]
        pts_depth = torch.from_numpy(pts_depth).float()
        
        zero_indices = (pts_depth == 0).nonzero(as_tuple=False)
        pts_depth[zero_indices[:, 0], zero_indices[:, 1], zero_indices[:, 2]] = 900
        gt_depth = visualize_depth(pts_depth)
        save_image(gt_depth, "gt_depth.png")
    
    save_image(gt_image, 'gt_image.png')
    save_image(dynamic_mask, 'mask.png')

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  cx=cx, cy=cy, fx=fx, fy=fy,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  dynamic_mask=dynamic_mask, pts_depth=pts_depth)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
