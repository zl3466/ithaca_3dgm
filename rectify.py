from ithaca365.ithaca365 import Ithaca365
import os
from PIL import Image
import numpy as np
import cv2
from ithaca365.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
import os.path as osp
from pyquaternion import Quaternion


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


def rectify_img(pointsensor, cam):
    # cam = nusc.get('sample_data', camera_token)
    # pointsensor = nusc.get('sample_data', pointsensor_token)
    # print(pc.points.shape)
    im = Image.open(osp.join(nusc.dataroot, cam['filename']))

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    intrinsic = np.array(cs_record['camera_intrinsic']).reshape((3, 3))
    distCoeff = np.array(cs_record['dist_coeff']).reshape(1, 5)
    R = np.array(cs_record['rectification_r']).reshape((3, 3))
    P = np.array(cs_record['rectification_p']).reshape((3, 4))
    print(im.size)
    width, height = im.size
    im = np.array(im)
    im = apply_rectify(intrinsic, distCoeff, R, P, (width, height), im)
    im = Image.fromarray(im)

    return im


def apply_rectify(instrinsic, distCoeff, R, P, size, img):
    mapx, mapy = cv2.initUndistortRectifyMap(
        cameraMatrix=instrinsic,
        distCoeffs=distCoeff,
        R=R,
        newCameraMatrix=P,
        size=size,
        m1type=cv2.CV_32FC1)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)


nusc = Ithaca365(version='v2.21', dataroot=r"/mnt/HDD/Ithaca", verbose=True)
# Pick a traversal
date2num = {'11-19-2021': 4, '11-22-2021': 5, '11-23-2021': 6, '11-29-2021': 7, '11-30-2021': 8, '12-01-2021': 9,
            '12-02-2021': 10, '12-03-2021': 11, '12-06-2021': 12, '12-07-2021': 13,
            '12-08-2021': 14, '12-09-2021': 15, '12-13-2021': 16, '12-14-2021': 17, '12-15-2021': 18, '12-16-2021': 19,
            '12-18-2021': 20, '12-19-2021': 21, '12-19-2021b': 22, '01-16-2022': 23}
length = 200
folder_path = "Ithaca365/loc" + str(2450) + "/input_rectify2/"
os.makedirs(folder_path, exist_ok=True)
for date in ['11-19-2021', '11-22-2021', '11-30-2021', '12-01-2021', '12-06-2021', '12-07-2021', '12-14-2021',
             '12-15-2021', '12-16-2021', '01-16-2022']:
    traversal_idx = date2num[date] - 1
    imgs2, lidar = extract_sample_data(2450, traversal_idx, date, length)
    imgs2_sub = imgs2[::2]
    lidar_sub = lidar[::2]
    for j in range(len(imgs2_sub)):
        cam_tk = imgs2_sub[j]
        pc_tk = lidar_sub[j]
        img = rectify_img(pc_tk, cam_tk)
        date_num = date.split('-')[0] + date.split('-')[1]
        loc_num = str(2450).zfill(4)
        img_num = str(j).zfill(3)
        cam_num = '2'
        new_name = date_num + loc_num + cam_num + img_num + '.png'

        # # Calculate the new size, half the original size
        # new_size = (img.width // 2, img.height // 2)

        # # Resize the image to the new size
        # downsampled_img = img.resize(new_size)

        # # Save the downsampled image to the specified output path
        # downsampled_img.save(folder_path +new_name)
        img.save(folder_path + new_name)
