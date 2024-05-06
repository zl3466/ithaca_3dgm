import json
import time

import torch
from torchvision.utils import save_image
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
import open3d as o3d
from matplotlib import cm
import matplotlib as mpl
import torchvision.transforms.functional as fn

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
        # print(cam2_data_my_traversal)
        # print("")

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


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, scale=1):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height * scale
        width = intr.width * scale

        uid = intr.id
        # R = np.transpose(qvec2rotmat(extr.qvec))
        R = np.array(extr.qvec)
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0] * scale
        elif intr.model == "PINHOLE":
            fx = intr.params[0] * scale
            fy = intr.params[1] * scale
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        # image_path = os.path.join(images_folder, os.path.basename(extr.name.split(".")[0] + ".jpg"))
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


def find_scale(img_list, loc, cam_infos):
    global_t_list = []
    global_r_list = []

    colmap_t_list = []
    colmap_r_list = []
    for i in range(len(img_list)):
        date = img_list[i][0]
        imgs2_sub = img_list[i][1]
        for j in range(len(imgs2_sub)):
            cam_tk = imgs2_sub[j]

            date_num = date.split('-')[0] + date.split('-')[1]
            loc_num = loc.zfill(4)
            img_num = str(j).zfill(3)
            cam_num = '2'
            new_name = date_num + loc_num + cam_num + img_num
            # print(cam_infos)
            # print(new_name)
            # print([cam_info for cam_info in cam_infos if cam_info.image_name == new_name])
            filtered_cam_infos = [cam_info for cam_info in cam_infos if cam_info.image_name == new_name][0]

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

    R_aligned, t_aligned, similarity_mat, s_opt = align_cameras(colmap_r_list, global_r_list, colmap_t_list,
                                                                global_t_list)

    return R_aligned, t_aligned, similarity_mat, s_opt.value


def depth_to_pcd(depth_map, scale, cam_intrinsic):
    # rgbd = o3d.geometry.RGBDImage()
    # rgbd = rgbd.create_from_color_and_depth(color=color_img, depth=depth_map, depth_scale=scale, convert_rgb_to_intensity=False)
    # pcd = o3d.geometry.PointCloud()
    # pcd = pcd.create_from_rgbd_image(rgbd, cam_intrinsic, cam_extrinsic)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth=depth_map, intrinsic=cam_intrinsic, depth_scale=scale)

    # flip the orientation, so it looks upright, not upside-down
    # pcd.transform([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return pcd


def visualize_depth(pts_depth, near=0.2, far=80, linear=False):
    pts_depth = torch.from_numpy(pts_depth).unsqueeze(0)
    zero_indices = (pts_depth == 0).nonzero(as_tuple=False)
    depth = pts_depth.clone()
    depth[zero_indices[:, 0], zero_indices[:, 1], zero_indices[:, 2]] = 900
    depth = depth[0].clone().detach().cpu().numpy()
    colormap = mpl.colormaps.get_cmap('turbo')
    # colormap = cm.get_cmap('gist_rainbow')
    curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    if linear:
        curve_fn = lambda x: -x
    eps = np.finfo(np.float32).eps
    near = near if near else depth.min()
    far = far if far else depth.max()
    near -= eps
    far += eps
    near, far, depth = [curve_fn(x) for x in [near, far, depth]]
    depth = np.nan_to_num(np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
    vis = colormap(depth)[:, :, :3]
    out_depth = np.clip(np.nan_to_num(vis), 0., 1.) * 255
    out_depth = torch.from_numpy(out_depth).permute(2, 0, 1).float().cuda() / 255
    # return out_depth.cpu().numpy()
    return out_depth


def generate_depth_visualize_pcd(pc_tk, cam_tk, cam_name, lidar_name, depth_name, scale, vis1, start_flag, loc):
    calib_lidar = nusc.get("calibrated_sensor", pc_tk['calibrated_sensor_token'])
    calib_cam = nusc.get("calibrated_sensor", cam_tk['calibrated_sensor_token'])
    cam_intrinsic = np.array(calib_cam["camera_matrix_rect"])
    cam_extrinsic = transform_matrix(np.array(calib_cam['translation_rect']),
                                     Quaternion(calib_cam['rotation_rect']).rotation_matrix)

    depth_img_name = depth_name.split(".")[0] + ".png"
    # image_data_path = os.path.join("I:/Ithaca365/loc2450/images", cam_name)
    # lidar_data_path = os.path.join("I:/Ithaca365/loc2450/lidar", lidar_name)
    # depth_data_path = os.path.join("I:/Ithaca365/loc2450/output/loc2450_segformer_idso_2/test/ours_30000/depth_map",
    #                                depth_name)
    # depth_viz_path = os.path.join("I:/Ithaca365/loc2450/output/loc2450_segformer_idso_2/test/ours_30000/depth_map_vis",
    #                               depth_img_name)
    image_data_path = os.path.join(f"I:/Ithaca365/data/{loc}/images", cam_name)
    pcl_path = os.path.join(f"I:/Ithaca365/data/{loc}/lidar", lidar_name)
    # depth_data_path = f"I:/Ithaca365/depth/{loc}_ours_30000/depth_map/{depth_name}"
    # depth_data_path = f"{root_dir}/gt_depth/{loc}/{cam_name.split('.')[0]}_depth_map.npy"
    depth_data_path = f"I:/Ithaca365/depth_sky_original_unit/{loc}/{depth_name.split('.')[0]}_depth_map.npy"
    scale = 1

    pc = LidarPointCloud.from_file(pcl_path)

    im = Image.open(image_data_path)
    im = im.resize((im.size[0] * 2, im.size[1] * 2))
    depth_3dgm = np.load(depth_data_path)
    depth_3dgm = depth_3dgm[0]
    depth_3dgm = cv2.resize(depth_3dgm, dsize=(im.size[0], im.size[1]), interpolation=cv2.INTER_CUBIC)
    depth_o3d = o3d.geometry.Image(depth_3dgm.astype(np.float32))
    fx = cam_intrinsic[0, 0]
    fy = cam_intrinsic[1, 1]
    cx = cam_intrinsic[0, 2]
    cy = cam_intrinsic[1, 2]
    # print(im.size[0], im.size[1], fx, fy, cx, cy)
    cam_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(im.size[0], im.size[1], fx, fy, cx, cy)

    depth_pcd = depth_to_pcd(depth_o3d, scale, cam_intrinsic_o3d)
    depth_pcd.scale(scale, (0, 0, 0))
    # depth_pcd = depth_pcd.voxel_down_sample(voxel_size=1.5)
    depth_pts = np.array(depth_pcd.points).T
    print(depth_pts.shape)

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

    depth_pts = depth_pts.T

    pts = pc.points[:3, :].T

    pts_color = np.array([[0, 0.5, 0.6]] * pts.shape[0])
    depth_pts_color = np.array([[0.6, 0.5, 0]] * depth_pts.shape[0])

    pcd_obj = o3d.geometry.PointCloud()
    pcd_obj.points = o3d.utility.Vector3dVector(np.concatenate((pts, depth_pts), axis=0))
    pcd_obj.colors = o3d.utility.Vector3dVector(np.concatenate((pts_color, depth_pts_color), axis=0))
    if start_flag:
        vis1.add_geometry(pcd_obj)
    else:
        while True:
            vis1.update_geometry(pcd_obj)
            vis1.poll_events()
            vis1.update_renderer()
            time.sleep(0.01)

    vis1.poll_events()
    vis1.update_renderer()
    time.sleep(0.01)
    return vis1


def generate_depth_original_size(root_dir, cam_name, depth_name, scale):
    new_name = cam_name.split(".")[0]
    loc_dir = f"{root_dir}/cam2_data/data/{loc}"
    img_dir = f"{loc_dir}/images/{cam_name}"
    seg_mask_dir = f"{loc_dir}/seg_mask"
    sky_mask_dir = f"{loc_dir}/sky_mask"
    # depth_dir = f"{root_dir}/cam2_data/depth_125/{loc}_ours_30000/depth_map/{depth_name}"
    depth_dir = f"{root_dir}/cam2_data/depth_125/{loc}_ours_30000/depth_map/{depth_name}"
    gt_dir = f"{root_dir}/cam2_data/gt_depth/{loc}/{cam_name.split('.')[0]}_depth_map.npy"
    output_dir = f"{loc_dir}/output"

    os.makedirs(output_dir, exist_ok=True)

    im = Image.open(img_dir)
    im = im.resize((im.size[0] * 2, im.size[1] * 2))
    depth_3dgm = np.load(depth_dir)
    # print(np.mean(depth_3dgm))
    depth_3dgm = depth_3dgm[0] * scale
    # print(np.mean(depth_3dgm))
    depth_3dgm = cv2.resize(depth_3dgm, dsize=(im.size[0], im.size[1]), interpolation=cv2.INTER_CUBIC)
    depth_gt = np.load(gt_dir)
    depth_gt = cv2.resize(depth_gt, dsize=(im.size[0], im.size[1]), interpolation=cv2.INTER_CUBIC)

    # cv2.imshow("gt1", depth_gt)
    # cv2.imshow("depth1", depth_3dgm / -scale)

    seg_mask = np.load(f"{seg_mask_dir}/{new_name}.npy")
    seg_mask = cv2.resize(seg_mask, dsize=(im.size[0], im.size[1]), interpolation=cv2.INTER_CUBIC)
    sky_mask = np.load(f"{sky_mask_dir}/{new_name}.npy")
    sky_mask = cv2.resize(sky_mask, dsize=(im.size[0], im.size[1]), interpolation=cv2.INTER_CUBIC)
    # mask = np.logical_and(seg_mask, sky_mask)

    # print(seg_mask)
    depth_gt_masked = np.where(seg_mask, depth_gt, 0)
    # depth_gt_masked = depth_gt

    ratio_sum = 0
    count = 0
    for i in range(depth_gt_masked.shape[0]):
        for j in range(depth_gt_masked.shape[1]):
            if depth_gt_masked[i, j] != 0:
                # print(f"{i}, {j}     lidar depth: {canvas[i, j]}, 3dgm depth: {depth_3dgm[i, j]}")
                txt_filename = f"test1.txt"
                f = open(txt_filename, "a")
                f.write(
                    f"{cam_name}    {i}, {j}     lidar depth: {depth_gt_masked[i, j]}, 3dgm depth: {depth_3dgm[i, j]}, ratio: {abs(depth_3dgm[i, j] - depth_gt_masked[i, j]) / depth_gt_masked[i, j]}" + "\n")
                f.close()
                ratio_sum += abs(depth_3dgm[i, j] - depth_gt_masked[i, j]) / depth_gt_masked[i, j]
                count += 1
    final_ratio = ratio_sum / count
    print(f"final ratio: {final_ratio}")

    # # save_path = f"{root_dir}/gt_depth_sky_original_unit/{loc}/{depth_name.split('.')[0]}_depth_map.npy"
    # # os.makedirs(f"{root_dir}/gt_depth_sky_original_unit/{loc}", exist_ok=True)
    # # np.save(save_path, depth_gt_masked)

    save_path = f"{root_dir}/cam2_data/gt_depth_original_unit_125/{loc}/{depth_name.split('.')[0]}_depth_map.npy"
    os.makedirs(f"{root_dir}/cam2_data/gt_depth_original_unit_125/{loc}", exist_ok=True)
    np.save(save_path, depth_gt_masked)

    save_path = f"{root_dir}/cam2_data/depth_sky_original_unit_125/{loc}/{depth_name.split('.')[0]}_depth_map.npy"
    os.makedirs(f"{root_dir}/cam2_data/depth_sky_original_unit_125/{loc}", exist_ok=True)
    np.save(save_path, np.array([depth_3dgm]))

    fig, ax = plt.subplots(1, 1, figsize=(9, 16))
    plt_canvas = fig.canvas
    ax.imshow(im)
    y = np.where(depth_gt_masked != 0)[0]
    x = np.where(depth_gt_masked != 0)[1]
    ax.scatter(x, y, c=depth_gt_masked[y, x], s=5)
    # ax.scatter(x_list, y_list, c=color_list, s=5)
    ax.axis('off')
    os.makedirs(f"{output_dir}/viz", exist_ok=True)
    fig.savefig(f"{output_dir}/viz/{cam_name}", bbox_inches='tight', pad_inches=0)

    depth_img_name = depth_name.split(".")[0] + ".png"
    # full_img2 = np.concatenate((cv2.merge((depth_gt_masked,depth_gt_masked,depth_gt_masked))*255, im), axis=1)
    # os.makedirs(f"{output_dir}/viz3", exist_ok=True)
    # cv2.imwrite(f"{output_dir}/viz3/{depth_img_name}", full_img2)
    # depth_gt_masked = (depth_gt_masked*255).astype(np.uint8)
    depth_gt_masked = visualize_depth(depth_gt_masked)
    depth_3dgm = visualize_depth(depth_3dgm)

    full_img = torch.cat((depth_gt_masked, depth_3dgm), 2)
    os.makedirs(f"{output_dir}/viz2", exist_ok=True)
    save_image(full_img, f"{output_dir}/viz2/{depth_img_name}")


def generate_depth_original_size_1000(root_dir, cam_name, depth_dir, scale):
    new_name = cam_name.split(".")[0]
    loc_dir = f"{root_dir}/cam2_data/data/{loc}"
    img_dir = f"{loc_dir}/images/{cam_name}"
    seg_mask_dir = f"{loc_dir}/seg_mask"
    sky_mask_dir = f"{loc_dir}/sky_mask"
    # depth_dir = f"{root_dir}/cam2_data/depth/{loc}_ours_30000/depth_map/{depth_name}"
    gt_dir = f"{root_dir}/cam2_data/gt_depth/{loc}/{cam_name.split('.')[0]}_depth_map.npy"
    output_dir = f"{loc_dir}/output"
    os.makedirs(output_dir, exist_ok=True)

    im = Image.open(img_dir)
    im = im.resize((im.size[0] * 2, im.size[1] * 2))
    depth_3dgm = np.load(depth_dir)
    # print(np.mean(depth_3dgm))
    depth_3dgm = depth_3dgm[0] * scale
    # print(np.mean(depth_3dgm))
    depth_3dgm = cv2.resize(depth_3dgm, dsize=(im.size[0], im.size[1]), interpolation=cv2.INTER_CUBIC)
    depth_gt = np.load(gt_dir)
    depth_gt = cv2.resize(depth_gt, dsize=(im.size[0], im.size[1]), interpolation=cv2.INTER_CUBIC)

    seg_mask = np.load(f"{seg_mask_dir}/{new_name}.npy")
    seg_mask = cv2.resize(seg_mask, dsize=(im.size[0], im.size[1]), interpolation=cv2.INTER_CUBIC)
    sky_mask = np.load(f"{sky_mask_dir}/{new_name}.npy")
    sky_mask = cv2.resize(sky_mask, dsize=(im.size[0], im.size[1]), interpolation=cv2.INTER_CUBIC)
    # mask = np.logical_and(seg_mask, sky_mask)
    # mask = seg_mask
    depth_gt_masked = np.where(seg_mask, depth_gt, 0)

    ''' save txt '''
    # ratio_sum = 0
    # count = 0
    # for i in range(depth_gt_masked.shape[0]):
    #     for j in range(depth_gt_masked.shape[1]):
    #         if depth_gt_masked[i, j] != 0:
    #             # print(f"{i}, {j}     lidar depth: {canvas[i, j]}, 3dgm depth: {depth_3dgm[i, j]}")
    #             # txt_filename = f"test1.txt"
    #             # f = open(txt_filename, "a")
    #             # f.write(f"{cam_name}    {i}, {j}     lidar depth: {depth_gt_masked[i, j]}, 3dgm depth: {depth_3dgm[i, j]}, ratio: {(depth_3dgm[i, j] - depth_gt_masked[i, j]) / depth_gt_masked[i, j]}" + "\n")
    #             # f.close()
    #             ratio_sum += abs(depth_3dgm[i, j] - depth_gt_masked[i, j]) / depth_gt_masked[i, j]
    #             count += 1
    # final_ratio = ratio_sum / count
    # print(f"final ratio: {final_ratio}")

    save_path = f"{root_dir}/cam2_data/gt_depth_original_unit_1000/{loc}/{new_name.split('.')[0]}_depth_map.npy"
    os.makedirs(f"{root_dir}/cam2_data/gt_depth_original_unit_1000/{loc}", exist_ok=True)
    np.save(save_path, depth_gt_masked)

    save_path = f"{root_dir}/cam2_data/depth_sky_original_unit_1000/{loc}/{new_name.split('.')[0]}_depth_map.npy"
    os.makedirs(f"{root_dir}/cam2_data/depth_sky_original_unit_1000/{loc}", exist_ok=True)
    np.save(save_path, np.array([depth_3dgm]))

    ''' visuializations '''
    ''' masked gt '''
    fig, ax = plt.subplots(1, 1, figsize=(9, 16))
    ax.imshow(im)
    y = np.where(depth_gt_masked != 0)[0]
    x = np.where(depth_gt_masked != 0)[1]
    ax.scatter(x, y, c=depth_gt_masked[y, x], s=5)
    ax.axis('off')
    os.makedirs(f"{output_dir}/gt_masked", exist_ok=True)
    fig.savefig(f"{output_dir}/gt_masked/{cam_name}", bbox_inches='tight', pad_inches=0)

    ''' original gt '''
    fig, ax = plt.subplots(1, 1, figsize=(9, 16))
    ax.imshow(im)
    y = np.where(depth_gt != 0)[0]
    x = np.where(depth_gt != 0)[1]
    ax.scatter(x, y, c=depth_gt[y, x], s=5)
    ax.axis('off')
    os.makedirs(f"{output_dir}/gt", exist_ok=True)
    fig.savefig(f"{output_dir}/gt/{cam_name}", bbox_inches='tight', pad_inches=0)

    ''' gt_masked + rendered depth '''
    depth_gt_masked = visualize_depth(depth_gt_masked)
    depth_3dgm = visualize_depth(depth_3dgm)

    full_img = torch.cat((depth_gt_masked, depth_3dgm), 2)
    os.makedirs(f"{output_dir}/{loc}/gt_rendered", exist_ok=True)
    save_image(full_img, f"{output_dir}/{loc}/gt_rendered/{cam_name}")


def generate_depth_original_size_depth_anything(pc_tk, cam_tk, folder_path, cam_name, lidar_name, depth_name, scale,
                                                seg_mask):
    calib_lidar = nusc.get("calibrated_sensor", pc_tk['calibrated_sensor_token'])
    calib_cam = nusc.get("calibrated_sensor", cam_tk['calibrated_sensor_token'])

    depth_img_name = "depth_" + cam_name

    image_data_path = os.path.join("I:/Ithaca365/loc2450/images", cam_name)
    lidar_data_path = os.path.join("I:/Ithaca365/loc2450/lidar", lidar_name)
    depth_data_path = os.path.join("I:/Ithaca365/loc2450/trial_result", depth_img_name.split(".")[0] + ".txt")
    depth_viz_path = os.path.join("I:/Ithaca365/loc2450/trial_result", depth_img_name)

    pc = LidarPointCloud.from_file(lidar_data_path)
    im = Image.open(image_data_path)
    depth_3dgm = np.loadtxt(depth_data_path)

    save_path = f"{folder_path}../depth_anything_original_unit/{depth_img_name.split('.')[0]}_depth_map.npy"
    depth_3dgm = cv2.resize(depth_3dgm, dsize=(im.size[0], im.size[1]), interpolation=cv2.INTER_CUBIC)
    if not os.path.exists(f"{folder_path}../depth_anything_original_unit"):
        os.makedirs(f"{folder_path}../depth_anything_original_unit")
    np.save(save_path, np.array([depth_3dgm]))
    depth_im = Image.open(depth_viz_path)
    depth_im = depth_im.resize((im.size[0], im.size[1]))

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

    pts = pc.points[:3, :].T

    depth_lidar = pts[:, 2]
    coloring = depth_lidar
    points = view_points(pts.T, np.array(calib_cam['camera_matrix_rect']), normalize=True)
    # points = view_points(pc.points[:3, :], np.array(calib_cam['camera_intrinsic']), normalize=True)

    mask = np.ones(depth_lidar.shape[0], dtype=bool)
    mask = np.logical_and(mask, depth_lidar > 1.0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    # print("target size", (depth_3dgm.shape[0], depth_3dgm.shape[1]))
    seg_mask = cv2.resize(seg_mask, dsize=(depth_3dgm.shape[1], depth_3dgm.shape[0]), interpolation=cv2.INTER_CUBIC)

    canvas = np.zeros(depth_3dgm.shape)
    for i in range(points.shape[1]):
        x = points[0, i]
        y = points[1, i]
        z = coloring[i]
        canvas[int(y), int(x)] = z

    x_list = []
    y_list = []
    color_list = []
    for i in range(points.shape[1]):
        x = points[0, i]
        y = points[1, i]
        z = coloring[i]
        if canvas[int(y), int(x)] != 0:
            x_list.append(x)
            y_list.append(y)
            color_list.append(z)

    fig, ax = plt.subplots(1, 1, figsize=(9, 16))
    plt_canvas = fig.canvas
    ax.imshow(im)
    ax.scatter(points[0, :], points[1, :], c=coloring, s=5)
    # ax.scatter(x_list, y_list, c=color_list, s=5)
    ax.axis('off')

    fig2, ax2 = plt.subplots(1, 1, figsize=(9, 16))
    plt_canvas2 = fig2.canvas
    ax2.imshow(depth_im)
    # ax.scatter(points[0, :], points[1, :], c=coloring, s=5)
    ax2.scatter(x_list, y_list, c=color_list, s=5)
    ax2.axis('off')
    print(f"original canvas: {canvas.shape}, 3dgm depth: {depth_3dgm.shape}")

    save_path = f"{folder_path}../gt_depth_original_unit/{depth_name.split('.')[0]}_depth_map.npy"
    if not os.path.exists(f"{folder_path}../gt_depth_original_unit"):
        os.makedirs(f"{folder_path}../gt_depth_original_unit")
    np.save(save_path, canvas)

    plt_canvas.draw()
    image_flat = np.frombuffer(plt_canvas.tostring_rgb(), dtype='uint8')
    image = image_flat.reshape(*reversed(plt_canvas.get_width_height()), 3)

    plt_canvas2.draw()
    image_flat2 = np.frombuffer(plt_canvas2.tostring_rgb(), dtype='uint8')
    image2 = image_flat2.reshape(*reversed(plt_canvas2.get_width_height()), 3)

    full_img = np.concatenate((image, image2), axis=1)
    cv2.imwrite(folder_path + depth_img_name, full_img)


def get_chamfer_distance(root_dir, pc_tk, cam_tk, cam_name, lidar_name, scale, depth_dir, vis1, start_flag):
    calib_lidar = nusc.get("calibrated_sensor", pc_tk['calibrated_sensor_token'])
    calib_cam = nusc.get("calibrated_sensor", cam_tk['calibrated_sensor_token'])
    loc_dir = f"{root_dir}/cam2_data/data/{loc}"
    img_dir = f"{loc_dir}/images/{cam_name}"
    lidar_dir = f"{loc_dir}/lidar/{lidar_name}"
    seg_mask_dir = f"{loc_dir}/seg_mask"

    cam_intrinsic = np.array(calib_cam["camera_matrix_rect"])
    cam_extrinsic = transform_matrix(np.array(calib_cam['translation_rect']),
                                     Quaternion(calib_cam['rotation_rect']).rotation_matrix)

    pc = LidarPointCloud.from_file(lidar_dir)
    im = Image.open(img_dir)
    im = im.resize((im.size[0] * 2, im.size[1] * 2))
    depth_3dgm = np.load(depth_dir)
    depth_3dgm = depth_3dgm[0]
    depth_3dgm = cv2.resize(depth_3dgm, dsize=(im.size[0], im.size[1]), interpolation=cv2.INTER_CUBIC)

    seg_mask = np.load(f"{seg_mask_dir}/{new_name}.npy")
    seg_mask = cv2.resize(seg_mask, dsize=(im.size[0], im.size[1]), interpolation=cv2.INTER_CUBIC)

    ''' depth image to pcd '''
    depth_o3d = o3d.geometry.Image(depth_3dgm.astype(np.float32))
    fx = cam_intrinsic[0, 0]
    fy = cam_intrinsic[1, 1]
    cx = cam_intrinsic[0, 2]
    cy = cam_intrinsic[1, 2]
    # print(im.size[0], im.size[1], fx, fy, cx, cy)
    cam_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(im.size[0], im.size[1], fx, fy, cx, cy)

    depth_pcd = depth_to_pcd(depth_o3d, scale, cam_intrinsic_o3d)
    depth_pcd.scale(scale, (0, 0, 0))
    depth_pts = np.array(depth_pcd.points).T

    ''' ==================== original pcd =================== '''
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

    pts = pc.points[:3, :].T

    points = view_points(pts.T, np.array(calib_cam['camera_matrix_rect']), normalize=True)
    # points = view_points(pc.points[:3, :], np.array(calib_cam['camera_intrinsic']), normalize=True)
    depth_lidar = pts[:, 2]
    coloring = depth_lidar
    mask = np.ones(depth_lidar.shape[0], dtype=bool)
    mask = np.logical_and(mask, depth_lidar > 1.0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    # points = points[:, mask]
    pts_new = pts.T[:, mask].T

    ''' apply seg_mask to pcd; remove points whose projection location has value zero in seg_mask '''
    # print(f"mask size: {np.where(mask == True)[0].shape}")
    # print(points.shape[1], mask.shape)
    for i in range(points.shape[1]):
        x = points[0, i]
        y = points[1, i]
        if 1 < x < im.size[0] - 1 and 1 < y < im.size[1] - 1:
            flag = seg_mask[int(y), int(x)]
            if flag == 0:
                mask[i] = False
    # print(f"combined mask: {np.where(mask == True)[0].shape}")
    pts_masked = pts.T[:, mask].T

    pcd_ori_masked = o3d.geometry.PointCloud()
    pcd_ori_masked.points = o3d.utility.Vector3dVector(pts_masked)

    pcd_ori = o3d.geometry.PointCloud()
    pcd_ori.points = o3d.utility.Vector3dVector(pts_new)

    # if start_flag:
    #     vis1.add_geometry(pcd_ori_masked)
    # else:
    #     while True:
    #         vis1.update_geometry(pcd_ori_masked)
    #         vis1.poll_events()
    #         vis1.update_renderer()
    #         time.sleep(0.01)
    #
    # vis1.poll_events()
    # vis1.update_renderer()
    # time.sleep(0.01)
    # return vis1

    ''' calculate different directions '''
    # dists1 = pcd_ori.compute_point_cloud_distance(depth_pcd)
    # dists2 = depth_pcd.compute_point_cloud_distance(pcd_ori)
    ''' gt to rendered '''
    # dists_masked = pcd_ori_masked.compute_point_cloud_distance(depth_pcd)
    # dists_masked = np.asarray(dists_masked)
    #
    # dists = pcd_ori.compute_point_cloud_distance(depth_pcd)
    # dists = np.asarray(dists)

    ''' rendered to gt '''
    dists_masked = depth_pcd.compute_point_cloud_distance(pcd_ori_masked)
    dists_masked = np.asarray(dists_masked)

    dists = depth_pcd.compute_point_cloud_distance(pcd_ori)
    dists = np.asarray(dists)

    # print(f"original: {np.mean(dists)}, masked: {np.mean(dists_masked)}")
    # print(pts.shape, dists.shape)
    return np.mean(dists), np.mean(dists_masked)


def get_chamfer_distance_depth_anything(pc_tk, cam_tk, cam_name, lidar_name, depth_name, scale, seg_mask, vis1,
                                        start_flag):
    calib_lidar = nusc.get("calibrated_sensor", pc_tk['calibrated_sensor_token'])
    calib_cam = nusc.get("calibrated_sensor", cam_tk['calibrated_sensor_token'])
    cam_intrinsic = np.array(calib_cam["camera_matrix_rect"])
    cam_extrinsic = transform_matrix(np.array(calib_cam['translation_rect']),
                                     Quaternion(calib_cam['rotation_rect']).rotation_matrix)

    depth_img_name = "depth_" + cam_name
    image_data_path = os.path.join("I:/Ithaca365/loc2450/images", cam_name)
    lidar_data_path = os.path.join("I:/Ithaca365/loc2450/lidar", lidar_name)
    depth_data_path = os.path.join("I:/Ithaca365/loc2450/trial_result", depth_img_name.split(".")[0] + ".txt")
    depth_viz_path = os.path.join("I:/Ithaca365/loc2450/trial_result", depth_img_name)

    pc = LidarPointCloud.from_file(lidar_data_path)
    im = Image.open(image_data_path)
    depth_3dgm = np.loadtxt(depth_data_path)
    depth_3dgm = cv2.resize(depth_3dgm, dsize=(im.size[0], im.size[1]), interpolation=cv2.INTER_CUBIC)
    depth_im = Image.open(depth_viz_path)
    depth_im = depth_im.resize((im.size[0], im.size[1]))

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

    pts = pc.points[:3, :].T

    points = view_points(pts.T, np.array(calib_cam['camera_matrix_rect']), normalize=True)
    # points = view_points(pc.points[:3, :], np.array(calib_cam['camera_intrinsic']), normalize=True)
    depth_lidar = pts[:, 2]
    coloring = depth_lidar
    mask = np.ones(depth_lidar.shape[0], dtype=bool)
    mask = np.logical_and(mask, depth_lidar > 1.0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    pts = pts.T[:, mask].T
    # seg_mask = cv2.resize(seg_mask, dsize=(depth_3dgm.shape[1], depth_3dgm.shape[0]), interpolation=cv2.INTER_CUBIC)
    # for i in range(points.shape[1]):
    #     x = points[0, i]
    #     y = points[1, i]
    #
    # count1 = len(np.where(canvas == 0)[0])
    # canvas = canvas * seg_mask
    canvas = np.zeros(depth_3dgm.shape)
    for i in range(points.shape[1]):
        x = points[0, i]
        y = points[1, i]
        z = coloring[i]
        canvas[int(y), int(x)] = 1

    # depth masked by ground truth (canvas)
    # depth_o3d = o3d.geometry.Image((depth_3dgm*canvas).astype(np.float32))
    depth_o3d = o3d.geometry.Image(depth_3dgm.astype(np.float32))
    fx = cam_intrinsic[0, 0]
    fy = cam_intrinsic[1, 1]
    cx = cam_intrinsic[0, 2]
    cy = cam_intrinsic[1, 2]
    # print(im.size[0], im.size[1], fx, fy, cx, cy)
    cam_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(im.size[0], im.size[1], fx, fy, cx, cy)

    depth_pcd = depth_to_pcd(depth_o3d, 1, cam_intrinsic_o3d)
    depth_pcd.scale(1, (0, 0, 0))
    # depth_pcd = depth_pcd.voxel_down_sample(voxel_size=1.5)
    depth_pts = np.array(depth_pcd.points).T
    print(depth_pts.shape)
    depth_pts = depth_pts.T

    pcd_ori = o3d.geometry.PointCloud()
    pcd_ori.points = o3d.utility.Vector3dVector(pts)

    # pts_color = np.array([[0, 0.5, 0.6]]*pts.shape[0])
    # depth_pts_color = np.array([[0.6, 0.5, 0]] * depth_pts.shape[0])

    # pcd_obj = o3d.geometry.PointCloud()
    # pcd_obj.points = o3d.utility.Vector3dVector(np.concatenate((pts, depth_pts), axis=0))
    # pcd_obj.colors = o3d.utility.Vector3dVector(np.concatenate((pts_color, depth_pts_color), axis=0))
    # if start_flag:
    #     vis1.add_geometry(pcd_obj)
    # else:
    #     while True:
    #         vis1.update_geometry(pcd_obj)
    #         vis1.poll_events()
    #         vis1.update_renderer()
    #         time.sleep(0.01)
    #
    # vis1.poll_events()
    # vis1.update_renderer()
    # time.sleep(0.01)
    # return vis1

    # dists1 = pcd_ori.compute_point_cloud_distance(depth_pcd)
    # dists2 = depth_pcd.compute_point_cloud_distance(pcd_ori)
    dists = pcd_ori.compute_point_cloud_distance(depth_pcd)
    dists = np.asarray(dists)
    print(pts.shape, dists.shape)
    return np.mean(dists)


def visualize_depth_original_size(root_dir, cam_name, depth_dir, scale, loc):
    loc_dir = f"{root_dir}/data/{loc}"
    img_dir = f"{root_dir}/images/{loc}/images/{cam_name}"

    img_cv2 = cv2.imread(img_dir)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    # img_cv2 = cv2.resize(img_cv2, dsize=(im.size[0], im.size[1]), interpolation=cv2.INTER_CUBIC)
    img_cv2 = torch.from_numpy(img_cv2).permute(2, 0, 1).float().cuda() / 255

    depth_3dgm = np.load(depth_dir)
    depth_3dgm = depth_3dgm[0] * -scale
    # depth_3dgm = cv2.resize(depth_3dgm, dsize=(im.size[0], im.size[1]), interpolation=cv2.INTER_CUBIC)

    depth_3dgm = visualize_depth(depth_3dgm)
    depth_3dgm = fn.resize(depth_3dgm, size=[img_cv2.shape[1], img_cv2.shape[2]])

    full_img = torch.cat((img_cv2, depth_3dgm), 2)
    output_dir = f"{loc_dir}/output/img_depth_viz_no_scale"
    os.makedirs(output_dir, exist_ok=True)
    save_image(full_img, f"{output_dir}/{cam_name}")

    # depth_3dgm = cv2.resize(depth_3dgm, dsize=(img_cv2.shape[1], img_cv2.shape[0]), interpolation=cv2.INTER_CUBIC)
    # full_img = np.concatenate((img_cv2, depth_3dgm), 1)
    # os.makedirs(f"{output_dir}/img_depth_viz", exist_ok=True)
    # cv2.imwrite(f"{output_dir}/img_depth_viz/{cam_name}", full_img)


def read_first_word(filename):
    try:
        with open(filename, 'r') as file:
            # Read the first line from the file
            first_line = file.readline().strip()
            # Split the line into words using space as delimiter
            words = first_line.split()
            # Return the first word
            return words[0] if words else None
    except FileNotFoundError:
        print("File not found.")

''' cf distance '''
# if __name__ == '__main__':
#     root_dir = "I:/Ithaca365"
#     # nusc = Ithaca365(version='v2.21', dataroot=r"/mnt/HDD/Ithaca", verbose=True)
#     nusc = Ithaca365(version='v2.21', dataroot=root_dir, verbose=True)
#     # Pick a traversal
#     date2num = {'11-19-2021': 4, '11-22-2021': 5, '11-23-2021': 6, '11-29-2021': 7, '11-30-2021': 8, '12-01-2021': 9,
#                 '12-02-2021': 10, '12-03-2021': 11, '12-06-2021': 12, '12-07-2021': 13,
#                 '12-08-2021': 14, '12-09-2021': 15, '12-13-2021': 16, '12-14-2021': 17, '12-15-2021': 18,
#                 '12-16-2021': 19,
#                 '12-18-2021': 20, '12-19-2021': 21, '12-19-2021b': 22, '01-16-2022': 23}
#     length = 200
#     cd_dict = {}
#     for loc in ["loc175", "loc550", "loc575", "loc600", "loc650", "loc750", "loc825", "loc975", "loc1200", "loc1500",
#                 "loc1650", "loc1700", "loc2200", "loc2300", "loc2350", "loc2400", "loc2450", "loc2500", "loc2525"]:
#         # if loc != "loc1200":
#         #     continue
#         ''' loc 1200 date 12-07, 14, 15, 16 very bad relative score '''
#
#         cameras_extrinsic_file = f"{root_dir}/cam2_data/data/{loc}/sparse/0/images.bin"
#         cameras_intrinsic_file = f"{root_dir}/cam2_data/data/{loc}/sparse/0/cameras.bin"
#
#         cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
#         cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
#         cam_infos = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
#                                       images_folder=f"{root_dir}/cam2_data/data/{loc}/images", scale=2)
#
#         scale = float(read_first_word(f"{root_dir}/cam2_data/data/{loc}/colmap_cam_pose/transform.txt"))
#         print(f"the overall scale for {loc} is {scale}")
#
#         vis1 = o3d.visualization.Visualizer()
#         vis1.create_window(
#             window_name='Ego Vehicle Segmented Scene',
#             width=256 * 4,
#             height=256 * 4,
#             left=480,
#             top=270)
#         vis1.get_render_option().background_color = [0, 0, 0]
#         vis1.get_render_option().point_size = 0.1
#         vis1.get_render_option().show_coordinate_frame = True
#
#         cd_mean_list = []
#         cd_mean_list_masked = []
#         test_idx = 0
#         train_idx = 0
#         counter = 0
#         # for date in ['11-19-2021', '11-22-2021', '11-30-2021', '12-01-2021', '12-06-2021', '12-07-2021',
#         #              '12-14-2021', '12-15-2021', '12-16-2021', '01-16-2022']:
#         for date in ['01-16-2022', '11-19-2021', '11-22-2021', '11-30-2021', '12-01-2021', '12-06-2021', '12-07-2021',
#                      '12-14-2021', '12-15-2021', '12-16-2021']:
#             start_flag = True
#             traversal_idx = date2num[date] - 1
#             imgs2, lidar = extract_sample_data(int(loc[3:]), traversal_idx, date, length)
#             imgs2_sub = imgs2[::2]
#             lidar_sub = lidar[::2]
#
#             global_cam_t = [0, 0, 0]
#             colmap_cam_t = [0, 0, 0]
#
#             img_idx = 0
#             cd_list = []
#             cd_list_masked = []
#             for i in tqdm(range(len(imgs2_sub)), desc=f"{loc}, {date}"):
#                 # for j in range(len(imgs2_sub)):
#                 if img_idx >= len(imgs2_sub):
#                     img_idx = img_idx - len(imgs2_sub)
#                     break
#                 cam_tk = imgs2_sub[img_idx]
#                 pc_tk = lidar_sub[img_idx]
#
#                 date_num = date.split('-')[0] + date.split('-')[1]
#                 loc_num = loc[3:].zfill(4)
#                 img_num = str(img_idx).zfill(3)
#                 cam_num = '2'
#                 # lidar_num = '0'
#                 new_name = date_num + loc_num + cam_num + img_num
#                 cam_name = new_name + ".jpg"
#                 lidar_name = date_num + loc_num + cam_num + img_num + ".bin"
#                 depth_name = '{0:05d}'.format(counter) + ".npy"
#
#                 filtered_cam_infos = [cam_info for cam_info in cam_infos if cam_info.image_name == new_name][0]
#
#                 # vis1 = generate_depth_visualize_pcd(pc_tk, cam_tk, cam_name, lidar_name, depth_name, scale, vis1, start_flag)
#
#                 # generate_depth_original_size_depth_anything(pc_tk, cam_tk, output_dir, cam_name, lidar_name, depth_name,
#                 #                                             scale, mask)
#
#                 # cd = get_chamfer_distance(pc_tk, cam_tk, cam_name, lidar_name, depth_name, scale, mask, vis1, start_flag)
#
#                 # cd = get_chamfer_distance_depth_anything(pc_tk, cam_tk, cam_name, lidar_name, depth_name, scale, mask, vis1, start_flag)
#
#                 # cd_list.append(cd)
#
#                 if counter % 8 == 0:
#                     depth_name = '{0:05d}'.format(test_idx) + ".npy"
#                     depth_dir = f"{root_dir}/cam2_data/cam2_down_seg+idso_depth/{loc}_test/depth_map/{depth_name}"
#                     test_idx += 1
#                 else:
#                     depth_name = '{0:05d}'.format(train_idx) + ".npy"
#                     depth_dir = f"{root_dir}/cam2_data/cam2_down_seg+idso_depth/{loc}_train/depth_map/{depth_name}"
#                     train_idx += 1
#
#                 # generate_depth_original_size_1000(root_dir, cam_name, depth_dir, scale)
#                 # generate_depth_original_size_1000_new(root_dir, cam_name, depth_dir, scale, pc_tk, cam_tk, lidar_name)
#                 cd, cd_masked = get_chamfer_distance(root_dir, pc_tk, cam_tk, cam_name, lidar_name, scale, depth_dir, vis1,
#                                           start_flag)
#
#                 cd_list.append(cd)
#                 cd_list_masked.append(cd_masked)
#                 print(f"{loc} {date} img{img_idx} cd: {cd}, cd_masked: {cd_masked}")
#
#                 # visualize_depth_original_size(root_dir, cam_name, depth_dir, scale, loc)
#                 start_flag = False
#                 img_idx += 1
#                 counter += 1
#
#             cd_list = np.array(cd_list)
#             cd_mean = np.mean(cd_list)
#             cd_list_masked = np.array(cd_list_masked)
#             cd_mean_masked = np.mean(cd_list_masked)
#             if loc not in cd_dict.keys():
#                 cd_dict[loc] = {date: [cd_mean, cd_mean_masked]}
#             else:
#                 cd_dict[loc][date] = [cd_mean, cd_mean_masked]
#             cd_mean_list.append(cd_mean)
#             cd_mean_list_masked.append(cd_mean_masked)
#
#         cd_mean_list = np.array(cd_mean_list)
#         final_mean = np.mean(cd_mean_list)
#         cd_mean_list_masked = np.array(cd_mean_list_masked)
#         final_mean_masked = np.mean(cd_mean_list_masked)
#         cd_dict[loc]["mean"] = [final_mean, final_mean_masked]
#         print(f"{loc} cd: {final_mean}, cd_masked: {final_mean_masked}")
#     with open("chamfer_distance_rendered_to_gt.json", 'w') as file:
#         json.dump(cd_dict, file, indent=4)

''' old main() '''
# if __name__ == '__main__':
#     root_dir = "I:/Ithaca365"
#     # nusc = Ithaca365(version='v2.21', dataroot=r"/mnt/HDD/Ithaca", verbose=True)
#     nusc = Ithaca365(version='v2.21', dataroot=root_dir, verbose=True)
#     # Pick a traversal
#     date2num = {'11-19-2021': 4, '11-22-2021': 5, '11-23-2021': 6, '11-29-2021': 7, '11-30-2021': 8, '12-01-2021': 9,
#                 '12-02-2021': 10, '12-03-2021': 11, '12-06-2021': 12, '12-07-2021': 13,
#                 '12-08-2021': 14, '12-09-2021': 15, '12-13-2021': 16, '12-14-2021': 17, '12-15-2021': 18,
#                 '12-16-2021': 19,
#                 '12-18-2021': 20, '12-19-2021': 21, '12-19-2021b': 22, '01-16-2022': 23}
#     length = 200
#     for loc in ["loc175", "loc550", "loc575", "loc600", "loc650", "loc750", "loc825", "loc975", "loc1200", "loc1500",
#                 "loc1650", "loc1700", "loc2200", "loc2300", "loc2350", "loc2400", "loc2450", "loc2500", "loc2525"]:
#         # if loc != "loc2450":
#         #     continue
#
#         cameras_extrinsic_file = f"{root_dir}/cam2_data/data/{loc}/sparse/0/images.bin"
#         cameras_intrinsic_file = f"{root_dir}/cam2_data/data/{loc}/sparse/0/cameras.bin"
#
#         cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
#         cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
#         cam_infos = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
#                                       images_folder=f"{root_dir}/cam2_data/data/{loc}/images", scale=2)
#
#         vis1 = o3d.visualization.Visualizer()
#         vis1.create_window(
#             window_name='Ego Vehicle Segmented Scene',
#             width=256 * 4,
#             height=256 * 4,
#             left=480,
#             top=270)
#         vis1.get_render_option().background_color = [0, 0, 0]
#         vis1.get_render_option().point_size = 0.1
#         vis1.get_render_option().show_coordinate_frame = True
#
#         depth_idx = 0
#         img_idx = 0
#
#         # img_list = []
#         # lidar_list = []
#         # for date in ['01-16-2022', '11-19-2021', '11-22-2021', '11-30-2021', '12-01-2021', '12-06-2021', '12-07-2021',
#         #              '12-14-2021', '12-15-2021', '12-16-2021']:
#         #     traversal_idx = date2num[date] - 1
#         #     imgs2, lidar = extract_sample_data(int(loc[3:]), traversal_idx, date, length)
#         #     imgs2_sub = imgs2[::2]
#         #     lidar_sub = lidar[::2]
#         #
#         #     img_list.append((date, imgs2_sub))
#         #     lidar_list.append((date, lidar_sub))
#         # print(img_list)
#         # R_aligned, t_aligned, similarity_mat, scale = find_scale(img_list, loc[3:], cam_infos)
#         scale = float(read_first_word(f"{root_dir}/cam2_data/data/{loc}/colmap_cam_pose/transform.txt"))
#         # print(f"the overall scale for {loc} is {scale}, colmap gives {new_scale}")
#         print(f"the overall scale for {loc} is {scale}")
#
#         cd_list = []
#         for date in ['01-16-2022', '11-19-2021', '11-22-2021', '11-30-2021', '12-01-2021', '12-06-2021', '12-07-2021',
#                      '12-14-2021', '12-15-2021', '12-16-2021']:
#             # if date != "12-06-2021":
#             #     continue
#             traversal_idx = date2num[date] - 1
#             imgs2, lidar = extract_sample_data(int(loc[3:]), traversal_idx, date, length)
#             imgs2_sub = imgs2[::2]
#             lidar_sub = lidar[::2]
#
#             global_cam_t = [0, 0, 0]
#             colmap_cam_t = [0, 0, 0]
#
#             start_flag = True
#             for i in tqdm(range(len(imgs2_sub)), desc=f"{date}, {loc}"):
#                 # for j in range(len(imgs2_sub)):
#                 if img_idx >= len(imgs2_sub):
#                     img_idx = img_idx - len(imgs2_sub)
#                     break
#                 cam_tk = imgs2_sub[img_idx]
#                 pc_tk = lidar_sub[img_idx]
#
#                 date_num = date.split('-')[0] + date.split('-')[1]
#                 loc_num = loc[3:].zfill(4)
#                 img_num = str(img_idx).zfill(3)
#                 cam_num = '2'
#                 lidar_num = '2'
#                 new_name = date_num + loc_num + cam_num + img_num
#                 cam_name = new_name + ".jpg"
#                 lidar_name = date_num + loc_num + lidar_num + img_num + ".bin"
#
#                 depth_name = '{0:05d}'.format(depth_idx) + ".npy"
#
#                 filtered_cam_infos = [cam_info for cam_info in cam_infos if cam_info.image_name == new_name][0]
#
#                 # vis1 = generate_depth_visualize_pcd(pc_tk, cam_tk, cam_name, lidar_name, depth_name, scale, vis1, start_flag, loc)
#
#                 generate_depth_original_size(root_dir, cam_name, depth_name, scale)
#
#
#                 # generate_depth_original_size_depth_anything(pc_tk, cam_tk, output_dir, cam_name, lidar_name, depth_name,
#                 #                                             scale, mask)
#
#                 # cd = get_chamfer_distance(pc_tk, cam_tk, cam_name, lidar_name, depth_name, scale, mask, vis1, start_flag)
#
#                 # cd = get_chamfer_distance_depth_anything(pc_tk, cam_tk, cam_name, lidar_name, depth_name, scale, mask, vis1, start_flag)
#
#                 # cd_list.append(cd)
#
#                 start_flag = False
#
#                 img_idx += 8
#                 depth_idx += 1
#
#         cd_list = np.array(cd_list)
#         print(cd_list.shape)
#         print(np.mean(cd_list))

if __name__ == '__main__':
    root_dir = "I:/Ithaca365"
    # nusc = Ithaca365(version='v2.21', dataroot=r"/mnt/HDD/Ithaca", verbose=True)
    nusc = Ithaca365(version='v2.21', dataroot=root_dir, verbose=True)
    # Pick a traversal
    date2num = {'11-19-2021': 4, '11-22-2021': 5, '11-23-2021': 6, '11-29-2021': 7, '11-30-2021': 8, '12-01-2021': 9,
                '12-02-2021': 10, '12-03-2021': 11, '12-06-2021': 12, '12-07-2021': 13,
                '12-08-2021': 14, '12-09-2021': 15, '12-13-2021': 16, '12-14-2021': 17, '12-15-2021': 18,
                '12-16-2021': 19,
                '12-18-2021': 20, '12-19-2021': 21, '12-19-2021b': 22, '01-16-2022': 23}
    length = 200
    cd_dict = {}
    for loc in ["loc175", "loc550", "loc575", "loc600", "loc650", "loc750", "loc825", "loc975", "loc1200", "loc1500",
                "loc1650", "loc1700", "loc2200", "loc2300", "loc2350", "loc2400", "loc2450", "loc2500", "loc2525"]:
        # if loc != "loc1200":
        #     continue
        ''' loc 1200 date 12-07, 14, 15, 16 very bad relative score '''

        cameras_extrinsic_file = f"{root_dir}/cam2_data/data/{loc}/sparse/0/images.bin"
        cameras_intrinsic_file = f"{root_dir}/cam2_data/data/{loc}/sparse/0/cameras.bin"

        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        cam_infos = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                      images_folder=f"{root_dir}/cam2_data/data/{loc}/images", scale=2)

        scale = float(read_first_word(f"{root_dir}/cam2_data/data/{loc}/colmap_cam_pose/transform.txt"))
        print(f"the overall scale for {loc} is {scale}")

        vis1 = o3d.visualization.Visualizer()
        vis1.create_window(
            window_name='Ego Vehicle Segmented Scene',
            width=256 * 4,
            height=256 * 4,
            left=480,
            top=270)
        vis1.get_render_option().background_color = [0, 0, 0]
        vis1.get_render_option().point_size = 0.1
        vis1.get_render_option().show_coordinate_frame = True


        test_idx = 0
        train_idx = 0
        counter = 0
        # for date in ['11-19-2021', '11-22-2021', '11-30-2021', '12-01-2021', '12-06-2021', '12-07-2021',
        #              '12-14-2021', '12-15-2021', '12-16-2021', '01-16-2022']:
        for date in ['01-16-2022', '11-19-2021', '11-22-2021', '11-30-2021', '12-01-2021', '12-06-2021', '12-07-2021',
                     '12-14-2021', '12-15-2021', '12-16-2021']:
            start_flag = True
            traversal_idx = date2num[date] - 1
            imgs2, lidar = extract_sample_data(int(loc[3:]), traversal_idx, date, length)
            imgs2_sub = imgs2[::2]
            lidar_sub = lidar[::2]

            global_cam_t = [0, 0, 0]
            colmap_cam_t = [0, 0, 0]

            img_idx = 0
            for i in tqdm(range(len(imgs2_sub)), desc=f"{loc}, {date}"):
                # for j in range(len(imgs2_sub)):
                if img_idx >= len(imgs2_sub):
                    img_idx = img_idx - len(imgs2_sub)
                    break
                cam_tk = imgs2_sub[img_idx]
                pc_tk = lidar_sub[img_idx]

                date_num = date.split('-')[0] + date.split('-')[1]
                loc_num = loc[3:].zfill(4)
                img_num = str(img_idx).zfill(3)
                cam_num = '2'
                # lidar_num = '0'
                new_name = date_num + loc_num + cam_num + img_num
                cam_name = new_name + ".jpg"
                lidar_name = date_num + loc_num + cam_num + img_num + ".bin"
                depth_name = '{0:05d}'.format(counter) + ".npy"

                filtered_cam_infos = [cam_info for cam_info in cam_infos if cam_info.image_name == new_name][0]

                if counter % 8 == 0:
                    depth_name = '{0:05d}'.format(test_idx) + ".npy"
                    depth_dir = f"{root_dir}/cam2_data/cam2_down_seg+idso_depth/{loc}_test/depth_map/{depth_name}"
                    test_idx += 1
                else:
                    depth_name = '{0:05d}'.format(train_idx) + ".npy"
                    depth_dir = f"{root_dir}/cam2_data/cam2_down_seg+idso_depth/{loc}_train/depth_map/{depth_name}"
                    train_idx += 1

                generate_depth_original_size_1000(root_dir, cam_name, depth_dir, scale)
                # generate_depth_original_size_1000_new(root_dir, cam_name, depth_dir, scale, pc_tk, cam_tk, lidar_name)

                # visualize_depth_original_size(root_dir, cam_name, depth_dir, scale, loc)

                img_idx += 1
                counter += 1





''' some commands '''
# zip -r images.zip loc1200/images loc1500/images loc1700/images loc1875/images loc2200/images loc2350/images loc2450/images loc2525/images loc550/images loc600/images loc750/images loc975/images loc1100/images loc1300/images loc1650/images loc175/images  loc1975/images loc2300/images loc2400/images loc2500/images loc275/images loc575/images loc650/images loc825/images

# scp yl7516@ai4ce-edison.poly.edu:/mnt/HDD/Ithaca/data/01-16-2022/decoded_lidar.tar.gz I:/
# scp yl7516@ai4ce-edison.poly.edu:/mnt/HDD/Ithaca/data/11-22-2021/decoded_lidar.tar.gz I:/
# scp yl7516@ai4ce-edison.poly.edu:/mnt/HDD/Ithaca/data/12-01-2021/decoded_lidar.tar.gz I:/
# scp yl7516@ai4ce-edison.poly.edu:/mnt/HDD/Ithaca/data/12-07-2021/decoded_lidar.tar.gz I:/
# scp yl7516@ai4ce-edison.poly.edu:/mnt/HDD/Ithaca/data/12-15-2021/decoded_lidar.tar.gz I:/
# scp yl7516@ai4ce-edison.poly.edu:/mnt/HDD/Ithaca/data/11-19-2021/decoded_lidar.tar.gz I:/
# scp yl7516@ai4ce-edison.poly.edu:/mnt/HDD/Ithaca/data/11-30-2021/decoded_lidar.tar.gz I:/
# scp yl7516@ai4ce-edison.poly.edu:/mnt/HDD/Ithaca/data/12-06-2021/decoded_lidar.tar.gz I:/
# scp yl7516@ai4ce-edison.poly.edu:/mnt/HDD/Ithaca/data/12-14-2021/decoded_lidar.tar.gz I:/
# scp yl7516@ai4ce-edison.poly.edu:/mnt/HDD/Ithaca/data/12-16-2021/decoded_lidar.tar.gz I:/

# zip -r lidar.zip 01-16-2022/decoded_lidar 11-22-2021/decoded_lidar 12-01-2021/decoded_lidar 12-07-2021/decoded_lidar  12-15-2021/decoded_lidar 11-19-2021/decoded_lidar 11-30-2021/decoded_lidar 12-06-2021/decoded_lidar 12-14-2021/decoded_lidar 12-16-2021/decoded_lidar

# the overall scale for loc175 is -20.78148902468401, colmap gives 20.861161783444754
# the overall scale for loc550 is -9.301939079207479, colmap gives 9.4460869731064125
# the overall scale for loc575 is -10.220767786288798, colmap gives 10.05713872756497
# the overall scale for loc600 is -16.993353493161937, colmap gives 16.961863309061229
# the overall scale for loc650 is -21.743636081048614, colmap gives 22.307181300297753
# the overall scale for loc750 is -17.751491942542163, colmap gives 17.897068659406784
# the overall scale for loc825 is -7.550936620412958, colmap gives 18.718694218604604
# the overall scale for loc975 is -14.771112895178414, colmap gives 14.665814564236038
# the overall scale for loc1200 is -34.95668310098159, colmap gives 41.564656087694885
# the overall scale for loc1500 is -25.43174598131795, colmap gives 28.881872492914383
# the overall scale for loc1650 is -24.880379641702714, colmap gives 24.988739991044042
# the overall scale for loc1700 is -24.459746197610674, colmap gives 24.3138911648395
# the overall scale for loc2200 is -9.780820771801075, colmap gives 9.5660105139085019
# the overall scale for loc2300 is -14.315864410252216, colmap gives 15.807043626869314
# the overall scale for loc2350 is -14.9234167994163, colmap gives 15.237457739744521
# the overall scale for loc2400 is -14.068021391475831, colmap gives 14.136399517464186
# the overall scale for loc2450 is -13.089235916937868, colmap gives 13.143354882160651
# the overall scale for loc2500 is -8.747474248354425, colmap gives 9.3322453899798443
# the overall scale for loc2525 is -13.764575416290876, colmap gives 14.102084638412393
