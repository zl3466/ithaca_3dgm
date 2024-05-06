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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import json
from tqdm import tqdm
from pyquaternion import Quaternion

CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
                    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                    'bicycle')

dynamic_objects = {'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'}

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    cx: float = None
    cy: float = None
    fx: float = None
    fy: float = None
    mask: np.array = None
    pointcloud_camera: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        print(T)  # check the position by colmap

        cx = None
        cy = None
        fx = None
        fy = None
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            fx = intr.params[0]
            fy = intr.params[1]
            cx = intr.params[2]
            cy = intr.params[3]
            #print("bra---------------")
            #print(intr.params)
            #print("ket---------------")
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        print(image_name)
        with open(os.path.join(images_folder.replace('images', 'meta'), 'camera_meta.json'), 'r') as file:
            pose_data = json.load(file)
        print('ground true camera pose is:', pose_data.get(image_name, 0)['camera_pose'], '\n')
        print('colmap camera pose is:', 'rotation',  extr.qvec, 'translation', extr.tvec)
        
        mask_path = os.path.join(images_folder.replace('images', 'images_sweeps'), os.path.basename(extr.name).replace('.png', '.npy'))
        mask = np.load(mask_path)
        
        world2cam = None
        
        lidar_name = os.path.basename(image_path).split(".")[0]
        lidar_name = lidar_name[:-4] + '0' + lidar_name[-3:] + '.bin'
        print(lidar_name)
        lidar_file = os.path.join(images_folder.replace('images', 'lidar'), lidar_name)
        scan = np.fromfile(lidar_file, dtype=np.float32)
        pc0 = scan.reshape((-1, 4))[:,:3]
        print(pc.shape)
        pc = np.concatenate([pc0, np.ones((pc.shape[0], 1))], axis=1)
        
        with open(os.path.join(images_folder.replace('images', 'meta'), 'lidar_meta.json'), 'r') as file:
            lidar_meta = json.load(file)
        print(lidar_meta['lidar_pose'])
        lidarR = Quaternion(lidar_meta['lidar_pose']['rotation']).rotation_matrix
        lidarT = np.array(lidar_meta['lidar_pose']['translation'])
        lidar2world = np.zeros((3, 4))
        lidar2world[:3, :3] = lidarR
        lidar2world[3, :3] = lidarT
        lidar2cam = world2cam @ lidar2world
        lidar_points = pc @ lidar2world.T  # this lidar_to_worlds
        point_camera = (np.pad(pc0, ((0, 0), (0, 1)), constant_values=1) @ lidar2cam.T)[:, :3]
        
        
        class_mapping = np.ones(len(CLASSES), dtype=np.int32)

        for i, cls in enumerate(CLASSES):
            if cls in dynamic_objects:
                class_mapping[i] = 0

        mask = class_mapping[mask][..., None]
        mask = np.repeat(mask, repeats=3, axis=-1)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, cx=cx, cy=cy, fx=fx, fy=fy, image=image, mask=mask,
                              image_path=image_path, image_name=image_name, width=width, height=height, pointcloud_camera=point_camera)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
    
def readManualSceneInfo(path, images, eval, llffhold=8):
    reading_dir = "images" if images == None else images
    image_folder = os.path.join(path, reading_dir)
    img_list = sorted(os.listdir(image_folder))
    
    cam_infos= []
    for idx, img_name in tqdm(enumerate(img_list)):
        if idx > 10:
            break
        image_path = os.path.join(image_folder, img_name)
        image = Image.open(image_path)
        width, height = image.size
        
        image_name = img_name.split('.')[0]
        lidar_name = image_name[:-4] + '0' + image_name[-3:]
        
        # process camera data------------------------------------------------------------------------------
        with open(os.path.join(path, 'meta', 'camera_meta.json'), 'r') as file:
            camera_meta = json.load(file)
        print('camera_timestamp: ', camera_meta[image_name]['timestamp'])
        camera2car = np.eye(4)
        camera2carR = Quaternion(camera_meta[image_name]['calib']['rotation_rect']).rotation_matrix
        camera2carT = np.array(camera_meta[image_name]['calib']['translation_rect'])
        camera2car[:3, :3] = camera2carR
        camera2car[:3, 3] = camera2carT
        
        car2world = np.eye(4)
        car2worldR = Quaternion(camera_meta[image_name]['camera_pose']['rotation']).rotation_matrix
        car2worldT = np.array(camera_meta[image_name]['camera_pose']['translation'])
        car2world[:3, :3] = car2worldR
        car2world[:3, 3] = car2worldT            
        
        camera2world = car2world @ camera2car
        world2camera = np.linalg.inv(camera2world)
        car2world1 = car2world
        R = world2camera[:3, :3].T  # WHETHER USE THIS T NEEDS CHECKS
        T = world2camera[:3, 3]
        
        # camera intrinsics-------------------------------------------------------------------------------
        intrinsic_matrix = camera_meta[image_name]['calib']['camera_matrix_rect']
        fx = intrinsic_matrix[0][0]
        cx = intrinsic_matrix[0][2]
        fy = intrinsic_matrix[1][1]
        cy = intrinsic_matrix[1][2]
        FovY = focal2fov(fy, height)
        FovX = focal2fov(fx, width)
        
        # handle mask--------------------------------------------------------------------------------------
        mask_path = os.path.join(path, 'images_sweeps', image_name + '.npy')
        mask = np.load(mask_path)
        
        class_mapping = np.ones(len(CLASSES), dtype=np.int32)

        for i, cls in enumerate(CLASSES):
            if cls in dynamic_objects:
                class_mapping[i] = 0

        mask = class_mapping[mask][..., None]
        mask = np.repeat(mask, repeats=3, axis=-1)
        
        # process lidar data-------------------------------------------------------------------------------
        lidar_file = os.path.join(path, 'lidar', lidar_name + '.bin')
        scan = np.fromfile(lidar_file, dtype=np.float32)
        pc0 = scan.reshape((-1, 4))[:,:3]
        
        with open(os.path.join(path, 'meta', 'lidar_meta.json'), 'r') as file:
            lidar_meta = json.load(file)
        print('lidar_timestamp: ', lidar_meta[lidar_name]['timestamp'])
        
        lidar2car = np.eye(4)
        lidar2carR = Quaternion(lidar_meta[lidar_name]['calib']['rotation']).rotation_matrix
        lidar2carT = np.array(lidar_meta[lidar_name]['calib']['translation'])
        lidar2car[:3, :3] = lidar2carR
        lidar2car[:3, 3] = lidar2carT
        
        car2world = np.eye(4)
        car2worldR = Quaternion(lidar_meta[lidar_name]['lidar_pose']['rotation']).rotation_matrix
        car2worldT = np.array(lidar_meta[lidar_name]['lidar_pose']['translation'])
        car2world[:3, :3] = car2worldR
        car2world[:3, 3] = car2worldT
        
        
        lidar2world = car2world @ lidar2car
        lidar2camera = world2camera @ lidar2world
        print('lidar2camera matrix:\n', lidar2camera)
        print('=============================================')

        point_camera = (np.pad(pc0, ((0, 0), (0, 1)), constant_values=1) @ lidar2camera.T)[:, :3]
        
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, cx=cx, cy=cy, fx=fx, fy=fy, image=image, mask=mask,
                              image_path=image_path, image_name=image_name, width=width, height=height, pointcloud_camera=point_camera)
        cam_infos.append(cam_info)
    
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    
    return scene_info


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    'Manual' : readManualSceneInfo,
}