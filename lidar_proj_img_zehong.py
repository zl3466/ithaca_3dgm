from ithaca365.ithaca365 import Ithaca365
import os
from PIL import Image
import numpy as np
import cv2
from ithaca365.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
import os.path as osp
from pyquaternion import Quaternion
import struct
import collections
import math
from typing import NamedTuple
import matplotlib.pyplot as plt
from ithaca365.utils.geometry_utils import view_points
import torch
import torch.nn.functional as F

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])

class Images(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    Fy: np.array
    Fx: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

def extract_sample_data(loc_idx, traversal_idx, date, length):
    my_location = nusc.location[loc_idx]
    # Load the camera tokens from all traversals of this location 
    cam2_data_all_traversal_my_location_token = nusc.query_by_location_and_channel(my_location['token'], 'cam2')
    cam2_data_my_traversal = nusc.get('sample_data', cam2_data_all_traversal_my_location_token[traversal_idx]) 

    lidar_data_all_traversal_my_location_token = nusc.query_by_location_and_channel(my_location['token'], 'LIDAR_TOP')
    lidar_data_my_traversal = nusc.get('sample_data', lidar_data_all_traversal_my_location_token[traversal_idx]) 

    imgs2 = []
    lidar = []
    # Load image paths composing a short sequence starting from this location
    while len(imgs2)<length:
        imgs2.append(cam2_data_my_traversal)
        lidar.append(lidar_data_my_traversal)

        cam2_data_my_traversal = nusc.get('sample_data', cam2_data_my_traversal['next']) 
        lidar_data_my_traversal = nusc.get('sample_data', lidar_data_my_traversal['next']) 

    return imgs2, lidar

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Images(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        # R = np.transpose(qvec2rotmat(extr.qvec))
        R = np.array(extr.qvec)
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
        elif intr.model=="PINHOLE":
            fx = intr.params[0]
            fy = intr.params[1]
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, Fx=fx, Fy=fy, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    return cam_infos

def trans_mat(calibration_lidar):
    translation_lidar = calibration_lidar['translation']
    rotation_quaternion_lidar = Quaternion(calibration_lidar['rotation'])
    rotation_matrix_lidar = rotation_quaternion_lidar.rotation_matrix
    T_LIDAR_to_Camera = np.eye(4)
    T_LIDAR_to_Camera[:3, :3] = rotation_matrix_lidar[:3, :3]
    T_LIDAR_to_Camera[:3, 3] = translation_lidar
    return T_LIDAR_to_Camera

def proj(pc_tk, cam_tk, folder_path, cam_name, lidar_name, new_name):
    calib_lidar = nusc.get("calibrated_sensor", pc_tk['calibrated_sensor_token'])
    calib_cam = nusc.get("calibrated_sensor", cam_tk['calibrated_sensor_token'])
    
    # image_data_path = os.path.join("/mnt/HDD/Ithaca365/loc2450/input_rectify2", cam_name)
    # lidar_data_path = os.path.join("/mnt/HDD/Ithaca365/loc2450/lidar", lidar_name)
    # pc = LidarPointCloud.from_file(lidar_data_path)
    pcl_path = osp.join(nusc.dataroot, pc_tk['filename'])
    pc = LidarPointCloud.from_file(pcl_path)
    # im = Image.open(image_data_path)
    im = Image.open(osp.join(nusc.dataroot, cam_tk['filename']))
    
    pc.rotate(Quaternion(calib_lidar['rotation']).rotation_matrix)
    pc.translate(np.array(calib_lidar['translation']))

    poserecord = nusc.get('ego_pose', pc_tk['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    poserecord = nusc.get('ego_pose', cam_tk['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    pc.translate(-np.array(calib_cam['translation']))
    pc.rotate(Quaternion(calib_cam['rotation']).rotation_matrix.T)

    # R = filtered_cam_infos.R
    # T = filtered_cam_infos.T
    # fx = filtered_cam_infos.Fx
    # fy = filtered_cam_infos.Fy
    # w, h = im.size
    # cx = w/2
    # cy = h/2
    # cam_intr = np.array([[fx, 0, cx],
    #           [0, fy, cy],
    #           [0, 0, 1]])

    # pc.translate(-T)
    # pc.rotate(Quaternion(R).rotation_matrix.T)

    depths = pc.points[2, :]
    coloring = depths
    points = view_points(pc.points[:3, :], np.array(calib_cam['camera_intrinsic']), normalize=True)

    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 1.0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    w, h = im.size
    depth_map = np.zeros((h, w))
    # depth_map = np.full((h, w), -np.inf, dtype=np.float32)
    for i in range(points.shape[1]):
        x, y = int(points[0, i]), int(points[1, i])
        depth_map[y, x] = coloring[i]

    # depth_map = torch.tensor(depth_map)
    # gt_depth_map = depth_map.unsqueeze(0).unsqueeze(0)
    # gt_depth_map = F.interpolate(gt_depth_map, size=(h//2, w//2), mode='bilinear',
    #                              align_corners=True)
    # gt_depth_map = gt_depth_map.squeeze(0).squeeze(0)
    # gt_depth_map = gt_depth_map.cpu().numpy()
    
    save_path = folder_path+new_name+'_depth_map.npy'
    np.save(save_path, depth_map)

    # fig, ax = plt.subplots(1, 1, figsize=(9, 16))
    # ax.imshow(im)
    # ax.scatter(points[0, :], points[1, :], c=coloring, s=5)
    # ax.axis('off')
    # fig.savefig(folder_path + cam_name, bbox_inches='tight', pad_inches=0)


nusc = Ithaca365(version='v2.21', dataroot=r"/mnt/HDD/Ithaca", verbose=True)
# Pick a traversal
date2num={'11-19-2021':4, '11-22-2021':5, '11-23-2021':6, '11-29-2021':7, '11-30-2021':8, '12-01-2021':9, '12-02-2021':10, '12-03-2021':11, '12-06-2021':12, '12-07-2021':13,
          '12-08-2021':14, '12-09-2021':15, '12-13-2021':16, '12-14-2021':17, '12-15-2021':18, '12-16-2021':19, '12-18-2021':20, '12-19-2021':21, '12-19-2021b':22, '01-16-2022':23}
length = 200
locations = [175, 550, 575, 600, 650, 750, 825, 975, 1200, 1500, 1650, 1700, 2200, 2300, 2350, 2400, 2450, 2500, 2525]
# locations = [1200]
for i in locations:
    # folder_path = "Ithaca365/loc" + str(i) + "/gt_depth/"
    folder_path = "gt_depth/loc" + str(i) + "/"
    os.makedirs(folder_path, exist_ok=True)
    cameras_extrinsic_file = os.path.join("/mnt/HDD/Ithaca365/loc2450", "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join("/mnt/HDD/Ithaca365/loc2450", "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    cam_infos = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder="/mnt/HDD/Ithaca365/loc2450/images")
    for date in ['11-19-2021', '11-22-2021', '11-30-2021', '12-01-2021', '12-06-2021', '12-07-2021', '12-14-2021', '12-15-2021', '12-16-2021', '01-16-2022']:
        traversal_idx = date2num[date]-1
        imgs2, lidar= extract_sample_data(i, traversal_idx, date, length)
        imgs2_sub = imgs2[::2]
        lidar_sub = lidar[::2]
        for j in range(len(imgs2_sub)):
            cam_tk = imgs2_sub[j]
            pc_tk = lidar_sub[j]

            date_num = date.split('-')[0]+date.split('-')[1]
            loc_num = str(i).zfill(4)
            img_num = str(j).zfill(3)
            cam_num = '2'
            lidar_num = '0'
            new_name = date_num + loc_num + cam_num + img_num
            cam_name =  new_name + ".png"
            lidar_name = date_num + loc_num + lidar_num + img_num + ".bin"

            # filtered_cam_infos = [cam_info for cam_info in cam_infos if cam_info.image_name == new_name][0]

            proj(pc_tk, cam_tk, folder_path, cam_name, lidar_name, new_name)
