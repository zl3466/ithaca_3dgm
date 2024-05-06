import numpy as np
from pyquaternion import Quaternion
import os
from ithaca365.ithaca365 import Ithaca365
from PIL import Image
from math import fabs, pi, atan2, asin, sin, cos
from typing import List


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


def transform_matrix(translation, rotation):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def rpy_to_quat(r, p, y):
    '''
    input: euler (degree)
    '''
    roll = r * np.pi / 180
    pitch = - p * np.pi / 180  # right-handed
    yaw = - y * np.pi / 180  # right-handed

    # nuScene coordinate system orientation as quaternion: w, x, y, z
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)

    return [qw, qx, qy, qz]


def quat_to_rpy(q: List[float]) -> List[float]:
    """
    Converts a May-style quaternion to a May-style roll-pitch-yaw value.

    q       - A quaternion, May-style, storing in format {w, x, y, z}
    returns - An array of three values representing, in order, roll, pitch, and yaw in radians.
    """
    qr = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    rpy = [0, 0, 0]

    disc = qr * qy - qx * qz

    if fabs(disc + 0.5) < 0.00000001:
        rpy[0] = 0
        rpy[1] = -pi / 2
        rpy[2] = 2 * atan2(qx, qr)
    elif fabs(disc - 0.5) < 0.0000001:
        rpy[0] = 0
        rpy[1] = pi / 2
        rpy[2] = -2 * atan2(qx, qr)
    else:
        # roll
        roll_a = 2 * (qr * qx + qy * qz)
        roll_b = 1 - 2 * (qx * qx + qy * qy)
        rpy[0] = atan2(roll_a, roll_b)

        # pitch
        rpy[1] = asin(2 * disc)

        # yaw
        yaw_a = 2 * (qr * qz + qx * qy)
        yaw_b = 1 - 2 * (qy * qy + qz * qz)
        rpy[2] = atan2(yaw_a, yaw_b)

    return rpy



def generate_colmap_cam_pose(img_id, cam_id, cam_model, cam_tk, output_dir):
    '''
    COLMAP: the X axis points to the right, the Y axis to the bottom, and the Z axis to the front
    需要绕x转90， 再绕新的y (老的z) 转90
    '''
    # rotx = Quaternion(axis=[0.0, 1.0, 0.0], degrees=90).rotation_matrix
    # tran1 = transform_matrix(np.array([0, 0, 0]), rotx)
    # rotz = Quaternion(axis=[1.0, 0.0, 0.0], degrees=-90).rotation_matrix
    # tran2 = transform_matrix(np.array([0, 0, 0]), rotz)
    # rotx = Quaternion(axis=[0.0, 1.0, 0.0], degrees=90).rotation_matrix
    # tran1 = transform_matrix(np.array([0, 0, 0]), rotx)
    # rotz = Quaternion(axis=[1.0, 0.0, 0.0], degrees=-90).rotation_matrix
    # tran2 = transform_matrix(np.array([0, 0, 0]), rotz)
    # tran = tran2.dot(tran1)

    # calib_lidar = nusc.get("calibrated_sensor", pc_tk['calibrated_sensor_token'])
    calib_cam = nusc.get("calibrated_sensor", cam_tk['calibrated_sensor_token'])
    cam_t = np.array(calib_cam['translation_rect'])
    cam_r = Quaternion(calib_cam['rotation_rect']).rotation_matrix
    cam_tran = transform_matrix(cam_t, cam_r)

    cam_poserecord = nusc.get('ego_pose', cam_tk['ego_pose_token'])
    cam_t_ego = np.array(cam_poserecord['translation'])
    cam_r_ego = Quaternion(cam_poserecord['rotation']).rotation_matrix
    cam_tran_ego = transform_matrix(cam_t_ego, cam_r_ego)

    cam_to_global = cam_tran_ego.dot(cam_tran)
    # cam_to_global = tran1.dot(cam_to_global)
    global_cam_r = cam_to_global[:3, :3]
    global_cam_t = cam_to_global[:3, 3]
    # global to cam
    # global_cam_r = cam_to_global[:3, :3].T
    # global_cam_t = -cam_to_global[:3, 3]

    # global_cam_quat = Quaternion(matrix=global_cam_r)
    # global_cam_eular = quat_to_rpy([global_cam_quat[0], global_cam_quat[1], global_cam_quat[2], global_cam_quat[3]])
    # print(global_cam_eular)
    #
    # global_cam_eular = [-global_cam_eular[1], -global_cam_eular[2], global_cam_eular[0]]
    # global_cam_quat = Quaternion(rpy_to_quat(global_cam_eular[0], global_cam_eular[1], global_cam_eular[2]))
    # global_cam_r = global_cam_quat.rotation_matrix
    #
    # global_cam_t = [-global_cam_t[1], -global_cam_t[2], global_cam_t[0]]

    image_data_path = os.path.join("I:/Ithaca365/loc2450/images", cam_name)
    im = Image.open(image_data_path)

    # Camera list with one line of data per camera:
    # CAMERA_ID, MODEL, WIDTH, HEIGHT, focal length, principal point x, principal point y
    intrinsic = np.array(calib_cam['camera_matrix_rect'])
    # print(calib_cam['camera_matrix_rect'])
    # print(calib_cam['camera_intrinsic'])
    intrinsic_text = f"{cam_id} {cam_model} {im.size[0]} {im.size[1]} {intrinsic[0, 0]} {intrinsic[1, 1]} {intrinsic[0, 2]} {intrinsic[1, 2]}"

    # Image list with two lines of data per image:
    # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    global_cam_r_quat = Quaternion(matrix=global_cam_r)
    extrinsic = [global_cam_r_quat[0], global_cam_r_quat[1], global_cam_r_quat[2], global_cam_r_quat[3], global_cam_t[0], global_cam_t[1], global_cam_t[2]]
    extrinsic_text = f"{img_id} " \
                     f"{global_cam_r_quat[0]} {global_cam_r_quat[1]} {global_cam_r_quat[2]} {global_cam_r_quat[3]} " \
                     f"{global_cam_t[0]} {global_cam_t[1]} {global_cam_t[2]} " \
                     f"{cam_id} {cam_name}"
    if not os.path.exists(f"{output_dir}/cameras.txt"):
        txt_filename = f"{output_dir}/cameras.txt"
        f = open(txt_filename, "a")
        f.write(intrinsic_text)
        f.close()

    txt_filename = f"{output_dir}/images.txt"
    f = open(txt_filename, "a")
    f.write(extrinsic_text + "\n" + "\n")
    f.close()



def generate_colmap_cam_extrinsic(cam_name, cam_tk, output_dir, flag):
    # calib_lidar = nusc.get("calibrated_sensor", pc_tk['calibrated_sensor_token'])
    calib_cam = nusc.get("calibrated_sensor", cam_tk['calibrated_sensor_token'])
    cam_t = np.array(calib_cam['translation_rect'])
    cam_r = Quaternion(calib_cam['rotation_rect']).rotation_matrix
    cam_tran = transform_matrix(cam_t, cam_r)

    cam_poserecord = nusc.get('ego_pose', cam_tk['ego_pose_token'])
    cam_t_ego = np.array(cam_poserecord['translation'])
    cam_r_ego = Quaternion(cam_poserecord['rotation']).rotation_matrix
    cam_tran_ego = transform_matrix(cam_t_ego, cam_r_ego)

    cam_to_global = cam_tran_ego.dot(cam_tran)
    # cam_to_global = tran1.dot(cam_to_global)

    global_cam_t = cam_to_global[:3, 3]

    # IMG_NAME, TX, TY, TZ
    extrinsic_text = f"{cam_name} {global_cam_t[0]} {global_cam_t[1]} {global_cam_t[2]} "
    txt_filename = f"{output_dir}/ref_pose.txt"
    if flag:
        f = open(txt_filename, "w")
        f.write(extrinsic_text + "\n")
        f.close()
    else:
        f = open(txt_filename, "a")
        f.write(extrinsic_text + "\n")
        f.close()




if __name__ == '__main__':
    root_dir = f"I:/Ithaca365"
    nusc = Ithaca365(version='v2.21', dataroot=root_dir, verbose=True)
    for loc in ["loc175", "loc550", "loc575", "loc600", "loc650", "loc750", "loc825", "loc975", "loc1200", "loc1500",
                "loc1650", "loc1700", "loc2200", "loc2300", "loc2350", "loc2400", "loc2450", "loc2500", "loc2525"]:
        # Pick a traversal
        date2num = {'11-19-2021': 4, '11-22-2021': 5, '11-23-2021': 6, '11-29-2021': 7, '11-30-2021': 8, '12-01-2021': 9,
                    '12-02-2021': 10, '12-03-2021': 11, '12-06-2021': 12, '12-07-2021': 13,
                    '12-08-2021': 14, '12-09-2021': 15, '12-13-2021': 16, '12-14-2021': 17, '12-15-2021': 18,
                    '12-16-2021': 19,
                    '12-18-2021': 20, '12-19-2021': 21, '12-19-2021b': 22, '01-16-2022': 23}
        length = 200

        output_dir = f"{root_dir}/data/{loc}/colmap_cam_pose"
        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}/transform.txt', 'w') as fp:
            pass

        img_id = 1
        rewrite_flag = True
        for date in ['11-19-2021', '11-22-2021', '11-30-2021', '12-01-2021', '12-06-2021', '12-07-2021', '12-14-2021',
                     '12-15-2021', '12-16-2021', '01-16-2022']:
            traversal_idx = date2num[date] - 1
            imgs2, lidar = extract_sample_data(int(loc[3:]), traversal_idx, date, length)
            imgs2_sub = imgs2[::2]
            lidar_sub = lidar[::2]

            # for j in tqdm(range(len(imgs2_sub)), desc=date):
            for j in range(len(imgs2_sub)):
                cam_tk = imgs2_sub[j]
                pc_tk = lidar_sub[j]

                date_num = date.split('-')[0] + date.split('-')[1]
                loc_num = str(loc[3:]).zfill(4)
                img_num = str(j).zfill(3)
                cam_num = '2'
                lidar_num = '0'
                new_name = date_num + loc_num + cam_num + img_num
                cam_name = new_name + ".jpg"
                lidar_name = date_num + loc_num + lidar_num + img_num + ".bin"

                # generate_colmap_cam_pose(img_id, "1", "PINHOLE", cam_tk, output_dir)
                generate_colmap_cam_extrinsic(cam_name, cam_tk, output_dir, rewrite_flag)
                if rewrite_flag:
                    rewrite_flag = False

                img_id += 1


    # .\COLMAP.bat feature_extractor --database_path I:/Ithaca365/loc2450/database.db --image_path I:/Ithaca365/loc2450/images
    # .\COLMAP.bat exhaustive_matcher --database_path I:/Ithaca365/loc2450/database.db
    #
    # .\COLMAP.bat point_triangulator --database_path I:/Ithaca365/loc2450/database.db --image_path I:/Ithaca365/loc2450/images --input_path I:/Ithaca365/loc2450/sparse/0 --output_path I:/Ithaca365/loc2450/sparse/0_0

    # scp yl7516@ai4ce-edison.poly.edu:/home/yl7516/dev/3DGM/output.zip I:/Ithaca365/loc2450

    # .\COLMAP.bat model_aligner --input_path I:/Ithaca365/data/loc175/sparse/0 --output_path I:/Ithaca365/data/loc175/sparse/1 --ref_images_path I:/Ithaca365/data/loc175/colmap_cam_pose/ref_pose.txt --ref_is_gps 0 --alignment_type ecef --robust_alignment 1 --robust_alignment_max_error 3.0

# .\COLMAP.bat model_aligner --input_path I:/Ithaca365/data/loc175/sparse/0 --output_path I:/Ithaca365/data/loc175/sparse/1 --ref_images_path I:/Ithaca365/data/loc175/colmap_cam_pose/ref_pose.txt --ref_is_gps 0 --robust_alignment 1 --robust_alignment_max_error 3.0

# .\COLMAP.bat model_aligner --input_path I:/Ithaca365/data/loc175/sparse/0 --output_path I:/Ithaca365/data/loc175/colmap_cam_pose --ref_images_path I:/Ithaca365/data/loc175/colmap_cam_pose/ref_pose.txt --ref_is_gps 0 --alignment_type custom  --alignment_max_error 3.0 --transform_path I:/Ithaca365/data/loc175/colmap_cam_pose/ref_pose.txt/transform.txt
