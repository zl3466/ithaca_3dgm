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
from ithaca365.utils.ithrectify import apply_rectify
from tqdm import tqdm
import cvxpy as cp

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
    while len(imgs2) < length:
        imgs2.append(cam2_data_my_traversal)
        lidar.append(lidar_data_my_traversal)

        cam2_data_my_traversal = nusc.get('sample_data', cam2_data_my_traversal['next'])
        lidar_data_my_traversal = nusc.get('sample_data', lidar_data_my_traversal['next'])

    return imgs2, lidar


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


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
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D,
                                       format_char_sequence="ddq" * num_points2D)
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
            params = read_next_bytes(fid, num_bytes=8 * num_params,
                                     format_char_sequence="d" * num_params)
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

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
        elif intr.model == "PINHOLE":
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


def transform_matrix(translation, rotation):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def proj2(pc_tk, cam_tk, filtered_cam_infos, folder_path, cam_name, lidar_name):
    calib_lidar = nusc.get("calibrated_sensor", pc_tk['calibrated_sensor_token'])
    calib_cam = nusc.get("calibrated_sensor", cam_tk['calibrated_sensor_token'])

    # cam_t = np.array(calib_cam['translation_rect'])
    # cam_r = Quaternion(calib_cam['rotation_rect']).rotation_matrix
    # cam_tran = transform_matrix(cam_t, cam_r)
    #
    # cam_poserecord = nusc.get('ego_pose', cam_tk['ego_pose_token'])
    # cam_t_ego = np.array(cam_poserecord['translation'])
    # cam_r_ego = Quaternion(cam_poserecord['rotation']).rotation_matrix
    # cam_tran_ego = transform_matrix(cam_t_ego, cam_r_ego)
    #
    # cam_to_global = cam_tran_ego.dot(cam_tran)
    # global_cam_r = cam_to_global[:3, :3]
    # global_cam_t = cam_to_global[:3, 3]

    # camera in colmap
    colmap_cam_t = np.array(filtered_cam_infos.T)
    colmap_cam_r = Quaternion(filtered_cam_infos.R).rotation_matrix

    # cam_name = cam_name[:7] + "00" + cam_name[9:]
    # image_data_path = os.path.join("I:/Ithaca365/loc2450/input", cam_name)
    image_data_path = os.path.join("I:/Ithaca365/loc2450/images", cam_name)
    lidar_data_path = os.path.join("I:/Ithaca365/loc2450/lidar", lidar_name)

    pc = LidarPointCloud.from_file(lidar_data_path)
    im = Image.open(image_data_path)
    # im = im.resize((im.size[0] * 2, im.size[1] * 2))
    # im = Image.open(osp.join(nusc.dataroot, cam_tk['filename']))

    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    pc.rotate(Quaternion(calib_lidar['rotation']).rotation_matrix)
    pc.translate(np.array(calib_lidar['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pc_tk['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    # poserecord = nusc.get('ego_pose', cam_tk['ego_pose_token'])
    # pc.translate(-np.array(poserecord['translation']))
    # pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)
    #
    # # Fourth step: transform from ego into the camera.
    # pc.translate(-np.array(calib_cam['translation_rect']))
    # pc.rotate(Quaternion(calib_cam['rotation_rect']).rotation_matrix.T)

    # pc.translate(-colmap_cam_t)
    # pc.rotate(colmap_cam_r.T)
    pc.translate(-colmap_cam_t)
    pc.rotate(colmap_cam_r.T)

    depths = pc.points[2, :]
    coloring = depths
    points = view_points(pc.points[:3, :], np.array(calib_cam['camera_matrix_rect']), normalize=True)
    # points = view_points(pc.points[:3, :], np.array(calib_cam['camera_intrinsic']), normalize=True)


    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 1.0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    fig, ax = plt.subplots(1, 1, figsize=(9, 16))
    ax.imshow(im)
    ax.scatter(points[0, :], points[1, :], c=coloring, s=5)
    ax.axis('off')
    # print(folder_path + cam_name)
    fig.savefig(folder_path + cam_name, bbox_inches='tight', pad_inches=0)
    print("===========================")



def proj3(pc_tk, cam_tk, filtered_cam_infos, folder_path, cam_name, lidar_name, prev_cam_t, prev_colmap_cam_t):
    calib_lidar = nusc.get("calibrated_sensor", pc_tk['calibrated_sensor_token'])
    calib_cam = nusc.get("calibrated_sensor", cam_tk['calibrated_sensor_token'])

    cam_t = np.array(calib_cam['translation_rect'])
    cam_r = Quaternion(calib_cam['rotation_rect']).rotation_matrix
    cam_tran = transform_matrix(cam_t, cam_r)

    cam_poserecord = nusc.get('ego_pose', cam_tk['ego_pose_token'])
    cam_t_ego = np.array(cam_poserecord['translation'])
    cam_r_ego = Quaternion(cam_poserecord['rotation']).rotation_matrix
    cam_tran_ego = transform_matrix(cam_t_ego, cam_r_ego)

    # original world to cam
    cam_to_global = cam_tran_ego.dot(cam_tran)
    global_cam_r = cam_to_global[:3, :3]
    global_cam_t = cam_to_global[:3, 3]

    # camera in colmap, world to cam
    colmap_cam_t = np.array(filtered_cam_infos.T)
    colmap_cam_r = Quaternion(filtered_cam_infos.R).rotation_matrix

    # rotation from colmap frame to global frame
    colmap_to_global_r = global_cam_r.dot(np.linalg.inv(colmap_cam_r))
    # rotate the colmap translation into global frame to help compare displacement along each axis
    # new_colmap_cam_t = colmap_to_global_r.dot(colmap_cam_t)

    global_delta_distance = np.sqrt((global_cam_t[0] - prev_cam_t[0]) ** 2 + (global_cam_t[1] - prev_cam_t[1]) ** 2 + (
                global_cam_t[2] - prev_cam_t[2]) ** 2)
    colmap_delta_distance = np.sqrt(
        (colmap_cam_t[0] - prev_colmap_cam_t[0]) ** 2 + (colmap_cam_t[1] - prev_colmap_cam_t[1]) ** 2 + (
                    colmap_cam_t[2] - prev_colmap_cam_t[2]) ** 2)
    scale = global_delta_distance / colmap_delta_distance
    print(f"distance: global {global_delta_distance}, colmap {colmap_delta_distance}, scale = {scale}")
    # print(f"distance each axis: global x: {global_delta_x} y: {global_delta_y} z: {global_delta_z}, colmap x: {colmap_delta_x} y: {colmap_delta_y} z: {colmap_delta_z}")
    # print(f"ratio for each axis: x {global_delta_x / colmap_delta_x}, y {global_delta_y / colmap_delta_y}, z {global_delta_z / colmap_delta_z}")

    # # if scale < 13.5:
    # #     scale = 13.5
    # scale = s_opt

    scale_matrix = np.array([[scale, 0, 0],
                             [0, scale, 0],
                             [0, 0, scale]])

    global_transform = np.zeros((4, 4))
    global_transform[:3, :3] = global_cam_r
    global_transform[:3, 3] = global_cam_t
    global_transform[3, 3] = 1
    colmap_transform = np.zeros((4, 4))
    new_colmap_r = scale_matrix.dot(colmap_cam_r)
    colmap_transform[:3, :3] = new_colmap_r
    colmap_transform[:3, 3] = colmap_cam_t * scale
    colmap_transform[3, 3] = 1

    # transform from global to colmap
    delta_transform = colmap_transform.dot(np.linalg.inv(global_transform))
    delta_r = new_colmap_r.dot(np.linalg.inv(global_cam_r))

    # cam_name = cam_name[:7] + "00" + cam_name[9:]
    # image_data_path = os.path.join("I:/Ithaca365/loc2450/input", cam_name)
    image_data_path = os.path.join("I:/Ithaca365/loc2450/images", cam_name)
    lidar_data_path = os.path.join("I:/Ithaca365/loc2450/lidar", lidar_name)

    pc = LidarPointCloud.from_file(lidar_data_path)
    im = Image.open(image_data_path)
    # im = im.resize((im.size[0] * 2, im.size[1] * 2))
    # im = Image.open(osp.join(nusc.dataroot, cam_tk['filename']))

    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    pc.rotate(Quaternion(calib_lidar['rotation']).rotation_matrix)
    pc.translate(np.array(calib_lidar['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pc_tk['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    # poserecord = nusc.get('ego_pose', cam_tk['ego_pose_token'])
    # pc.translate(-np.array(poserecord['translation']))
    # pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)
    #
    # # Fourth step: transform from ego into the camera.
    # pc.translate(-np.array(calib_cam['translation']))
    # pc.rotate(Quaternion(calib_cam['rotation']).rotation_matrix.T)

    # # fifth step: transform from the original global frame to the new colmap global frame
    # # pc.rotate(global_delta_cam_r.rotation_matrix)
    # # pc.translate(global_delta_cam_t)

    # global to colmap
    ''' from original world to colmap world '''
    pc.transform(delta_transform)

    # # colmap to colmap camera
    pc.translate(colmap_cam_t)
    pc.rotate(new_colmap_r)

    # pc.translate(-new_colmap_r.T.dot(colmap_cam_t * scale))
    # pc.rotate(new_colmap_r.T)

    depths = pc.points[2, :]
    coloring = depths
    # points = view_points(pc.points[:3, :], np.array(calib_cam['camera_matrix_rect']), normalize=True)
    # points = view_points(pc.points[:3, :], np.array(calib_cam['camera_intrinsic']), normalize=True)

    # intrinsics = np.array(calib_cam['camera_matrix_rect'])
    # intrinsics[0, 0] = filtered_cam_infos.Fx*2
    # intrinsics[1, 1] = filtered_cam_infos.Fy * 2
    #
    # intrinsics = np.array([844.75411541900951, 842.49633651533554, 480, 302])
    intrinsics = np.array([[844.75411541900951*2, 0, 480*2], [0, 842.49633651533554*2, 302*2], [0, 0, 1]])
    points = view_points(pc.points[:3, :], intrinsics, normalize=True)

    # print(calib_cam['camera_intrinsic'])
    # print(f"fx: {filtered_cam_infos.Fx*2}, fy: {filtered_cam_infos.Fy*2}, {im.size[0]/2}, {im.size[1]/2}")

    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 1.0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    fig, ax = plt.subplots(1, 1, figsize=(9, 16))
    ax.imshow(im)
    ax.scatter(points[0, :], points[1, :], c=coloring, s=5)
    ax.axis('off')
    # print(folder_path + cam_name)
    fig.savefig(folder_path + cam_name, bbox_inches='tight', pad_inches=0)
    print("===========================")

    return global_cam_t, colmap_cam_t


def proj(pc_tk, cam_tk, filtered_cam_infos, folder_path, cam_name, lidar_name):
    calib_lidar = nusc.get("calibrated_sensor", pc_tk['calibrated_sensor_token'])
    calib_cam = nusc.get("calibrated_sensor", cam_tk['calibrated_sensor_token'])

    # image_data_path = os.path.join("I:/Ithaca365/loc2450/input_rectify2", cam_name)
    # cam_name = cam_name[:7] + "00" + cam_name[9:]
    # image_data_path = os.path.join("I:/Ithaca365/loc2450/input", cam_name)
    image_data_path = os.path.join("I:/Ithaca365/loc2450/images", cam_name)
    lidar_data_path = os.path.join("I:/Ithaca365/loc2450/lidar", lidar_name)

    pc = LidarPointCloud.from_file(lidar_data_path)
    im = Image.open(image_data_path)
    # im = im.resize((im.size[0] * 2, im.size[1] * 2))

    ''' ======================================= '''
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    pc.rotate(Quaternion(calib_lidar['rotation']).rotation_matrix)
    pc.translate(np.array(calib_lidar['translation']))

    # # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pc_tk['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam_tk['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    ''' ================ if img already rectified ================ '''
    pc.translate(-np.array(calib_cam['translation_rect']))
    pc.rotate(Quaternion(calib_cam['rotation_rect']).rotation_matrix.T)

    ''' ======================================= '''

    # ''' ================ if use distorted img ================ '''
    # pc.translate(-np.array(calib_cam['translation']))
    # pc.rotate(Quaternion(calib_cam['rotation']).rotation_matrix.T)

    # ''' ================ if rectify now ================ '''
    # pc.translate(-np.array(calib_cam['translation_rect']))
    # pc.rotate(Quaternion(calib_cam['rotation_rect']).rotation_matrix.T)
    # intrinsic = np.array(calib_cam['camera_intrinsic']).reshape((3, 3))
    # distCoeff = np.array(calib_cam['dist_coeff']).reshape(1, 5)
    # R = np.array(calib_cam['rectification_r']).reshape((3, 3))
    # P = np.array(calib_cam['rectification_p']).reshape((3, 4))
    # print(im.size)
    # width, height = im.size
    # im = np.array(im)
    # im = apply_rectify(intrinsic, distCoeff, R, P, (width, height), im)
    # im = Image.fromarray(im)

    depths = pc.points[2, :]
    coloring = depths
    points = view_points(pc.points[:3, :], np.array(calib_cam['camera_matrix_rect']), normalize=True)
    # points = view_points(pc.points[:3, :], np.array(calib_cam['camera_intrinsic']), normalize=True)

    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 1.0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    fig, ax = plt.subplots(1, 1, figsize=(9, 16))
    ax.imshow(im)
    ax.scatter(points[0, :], points[1, :], c=coloring, s=5)
    ax.axis('off')
    # print(folder_path + cam_name)
    fig.savefig(folder_path + cam_name, bbox_inches='tight', pad_inches=0)


def align_cameras(pred_Rs, gt_Rs, pred_ts, gt_ts):
    '''
    :param pred_Rs: ndarray double - n x 3 x 3 predicted camera rotation
    :param gt_Rs: ndarray double - n x 3 x 3 camera ground truth rotation
    :param pred_ts: ndarray double - n x 3 predicted translation
    :param gt_ts: ndarray double - n x 3 ground truth translation
    :return:
    '''

    n = pred_Rs.shape[0]  # num of cameras

    # find rotation
    Q = np.sum(gt_Rs @ np.transpose(pred_Rs, [0, 2, 1]), axis=0)
    Uq, _, Vqh = np.linalg.svd(Q)
    sv = np.ones(3)
    sv[-1] = np.linalg.det(Uq @ Vqh)
    R_opt = Uq @ np.diag(sv) @ Vqh
    R_aligned = R_opt.reshape([1, 3, 3]) @ pred_Rs

    # find translation
    pred_ts = pred_ts @ R_opt.T  # Apply the optimal rotation on all the translations
    s_opt = cp.Variable()
    t_opt = cp.Variable((1, 3))
    constraints = []
    obj = cp.Minimize(cp.sum(cp.norm(gt_ts - (s_opt * pred_ts + np.ones((n, 1), dtype=np.double) @ t_opt), axis=1)))
    prob = cp.Problem(obj, constraints)
    prob.solve()
    t_aligned = s_opt.value * pred_ts + t_opt.value.reshape([1, 3])

    # get similarity matrix
    similarity_mat = np.eye(4)
    similarity_mat[0:3, 0:3] = s_opt.value * R_opt
    similarity_mat[0:3, 3] = t_opt.value

    return R_aligned, t_aligned, similarity_mat, s_opt



def find_scale(imgs2_sub, lidar_sub, cam_infos):
    global_t_list = []
    global_r_list = []

    colmap_t_list = []
    colmap_r_list = []

    for j in range(len(imgs2_sub)):
        cam_tk = imgs2_sub[j]
        pc_tk = lidar_sub[j]

        date_num = date.split('-')[0] + date.split('-')[1]
        loc_num = str(2450).zfill(4)
        img_num = str(j).zfill(3)
        cam_num = '2'
        lidar_num = '0'
        new_name = date_num + loc_num + cam_num + img_num
        cam_name = new_name + ".png"
        lidar_name = date_num + loc_num + lidar_num + img_num + ".bin"

        filtered_cam_infos = [cam_info for cam_info in cam_infos if cam_info.image_name == new_name][0]

        calib_lidar = nusc.get("calibrated_sensor", pc_tk['calibrated_sensor_token'])
        calib_cam = nusc.get("calibrated_sensor", cam_tk['calibrated_sensor_token'])

        cam_t = np.array(calib_cam['translation_rect'])
        cam_r = Quaternion(calib_cam['rotation_rect']).rotation_matrix
        cam_tran = transform_matrix(cam_t, cam_r)

        cam_poserecord = nusc.get('ego_pose', cam_tk['ego_pose_token'])
        cam_t_ego = np.array(cam_poserecord['translation'])
        cam_r_ego = Quaternion(cam_poserecord['rotation']).rotation_matrix
        cam_tran_ego = transform_matrix(cam_t_ego, cam_r_ego)

        # original world to cam
        cam_to_global = cam_tran_ego.dot(cam_tran)
        global_cam_r = cam_to_global[:3, :3]
        global_cam_t = cam_to_global[:3, 3]

        # camera in colmap, world to cam
        colmap_cam_t = np.array(filtered_cam_infos.T)
        colmap_cam_r = Quaternion(filtered_cam_infos.R).rotation_matrix

        global_t_list.append(global_cam_t)
        global_r_list.append(global_cam_r)
        colmap_t_list.append(colmap_cam_t)
        colmap_r_list.append(colmap_cam_r)

    global_t_list = np.array(global_t_list)
    global_r_list = np.array(global_r_list)
    colmap_t_list = np.array(colmap_t_list)
    colmap_r_list = np.array(colmap_r_list)

    R_aligned, t_aligned, similarity_mat, s_opt = align_cameras(colmap_r_list, global_r_list, colmap_t_list, global_t_list)

    print(f"scale = {s_opt.value}")




root_dir = "I:/Ithaca365/loc2450"
# nusc = Ithaca365(version='v2.21', dataroot=r"/mnt/HDD/Ithaca", verbose=True)
nusc = Ithaca365(version='v2.21', dataroot=root_dir, verbose=True)
# Pick a traversal
date2num = {'11-19-2021': 4, '11-22-2021': 5, '11-23-2021': 6, '11-29-2021': 7, '11-30-2021': 8, '12-01-2021': 9,
            '12-02-2021': 10, '12-03-2021': 11, '12-06-2021': 12, '12-07-2021': 13,
            '12-08-2021': 14, '12-09-2021': 15, '12-13-2021': 16, '12-14-2021': 17, '12-15-2021': 18, '12-16-2021': 19,
            '12-18-2021': 20, '12-19-2021': 21, '12-19-2021b': 22, '01-16-2022': 23}
length = 200
# folder_path = "Ithaca365/loc" + str(2450) + "/proj2/"
folder_path = f"{root_dir}/proj_scale/"
os.makedirs(folder_path, exist_ok=True)

cameras_extrinsic_file = f"{root_dir}/sparse_old/0/images.bin"
cameras_intrinsic_file = f"{root_dir}/sparse_old/0/cameras.bin"
cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
cam_infos = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                              images_folder=f"{root_dir}/images")

print(cam_intrinsics)
for date in ['11-19-2021', '11-22-2021', '11-30-2021', '12-01-2021', '12-06-2021', '12-07-2021', '12-14-2021',
             '12-15-2021', '12-16-2021', '01-16-2022']:
    traversal_idx = date2num[date] - 1
    imgs2, lidar = extract_sample_data(2450, traversal_idx, date, length)
    imgs2_sub = imgs2[::2]
    lidar_sub = lidar[::2]

    global_cam_t = [0, 0, 0]
    colmap_cam_t = [0, 0, 0]

    # find_scale(imgs2_sub, lidar_sub, cam_infos)

    for j in tqdm(range(len(imgs2_sub)), desc=date):
    # for j in range(len(imgs2_sub)):
        cam_tk = imgs2_sub[j]
        pc_tk = lidar_sub[j]

        date_num = date.split('-')[0] + date.split('-')[1]
        loc_num = str(2450).zfill(4)
        img_num = str(j).zfill(3)
        cam_num = '2'
        lidar_num = '0'
        new_name = date_num + loc_num + cam_num + img_num
        cam_name = new_name + ".png"
        lidar_name = date_num + loc_num + lidar_num + img_num + ".bin"

        filtered_cam_infos = [cam_info for cam_info in cam_infos if cam_info.image_name == new_name][0]

        proj(pc_tk, cam_tk, filtered_cam_infos, folder_path, cam_name, lidar_name)

        # proj2(pc_tk, cam_tk, filtered_cam_infos, folder_path, cam_name, lidar_name)

        # global_cam_t, colmap_cam_t = proj3(pc_tk, cam_tk, filtered_cam_infos, folder_path, cam_name, lidar_name,
        #                                    global_cam_t, colmap_cam_t)

        # nusc.render_pointcloud_in_image(cam_tk['sample_token'], pointsensor_channel='LIDAR_TOP',
        #                                 camera_channel='cam2', out_path=f"{root_dir}/proj/{cam_name}", rectify=True,
        #                                 cam_filename=cam_name,
        #                                 lidar_filename=lidar_name)


