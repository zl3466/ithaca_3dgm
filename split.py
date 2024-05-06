import shutil
import os
import imageio
from PIL import Image
import numpy as np
# import natsort 

from ithaca365.ithaca365 import Ithaca365
from tqdm import tqdm

dataset = Ithaca365(version="v2.21", dataroot="I:/Ithaca365", verbose=True)
dataset.list_scenes()


def extract_path(loc_idx, traversal_idx, date, length):
    my_location = dataset.location[loc_idx]
    # Load the camera tokens from all traversals of this location 
    cam0_data_all_traversal_my_location_token = dataset.query_by_location_and_channel(my_location['token'], 'LIDAR_TOP')
    cam0_data_my_traversal = dataset.get('sample_data', cam0_data_all_traversal_my_location_token[traversal_idx])

    # cam2_data_all_traversal_my_location_token = dataset.query_by_location_and_channel(my_location['token'], 'cam2')
    # cam2_data_my_traversal = dataset.get('sample_data', cam2_data_all_traversal_my_location_token[traversal_idx]) 

    # lidar_data_all_traversal_my_location_token = dataset.query_by_location_and_channel(my_location['token'], 'LIDAR_TOP')
    # lidar_data_my_traversal = dataset.get('sample_data', lidar_data_all_traversal_my_location_token[traversal_idx]) 

    # Find the initial index
    cam0_data_init_path = cam0_data_my_traversal['filename']
    cam0_data_init_num = cam0_data_init_path.split('/')[-1]
    img0_list = os.listdir('I:/Ithaca365/lidar/' + date + '/decoded_lidar')
    init_idx = img0_list.index(cam0_data_init_num)

    imgs0_path = []
    imgs2_path = []
    lidar_path = []
    # Load image paths composing a short sequence starting from this location
    while len(imgs0_path) < length:
        cam0_data_path = cam0_data_my_traversal['filename']
        filename = cam0_data_path.split("/")[-1]
        cam0_data_path = f'I:/Ithaca365/lidar/{date}/decoded_lidar/{filename}'
        imgs0_path.append(cam0_data_path)

        # cam2_data_path =  cam2_data_my_traversal['filename']
        # cam2_data_path = '/mnt/hdd_0/yl7516/ithaca/' + cam2_data_path
        # imgs2_path.append(cam2_data_path)

        # lidar_data_path =  lidar_data_my_traversal['filename']
        # lidar_data_path = '/mnt/hdd_0/yl7516/ithaca/' + lidar_data_path
        # lidar_path.append(lidar_data_path)

        cam0_data_my_traversal = dataset.get('sample_data', cam0_data_my_traversal['next'])
        # cam2_data_my_traversal = dataset.get('sample_data', cam2_data_my_traversal['next']) 
        # lidar_data_my_traversal = dataset.get('sample_data', lidar_data_my_traversal['next']) 

    return imgs0_path, imgs2_path, lidar_path


# Pick a traversal
date2num = {'11-19-2021': 4, '11-22-2021': 5, '11-23-2021': 6, '11-29-2021': 7, '11-30-2021': 8, '12-01-2021': 9,
            '12-02-2021': 10, '12-03-2021': 11, '12-06-2021': 12, '12-07-2021': 13,
            '12-08-2021': 14, '12-09-2021': 15, '12-13-2021': 16, '12-14-2021': 17, '12-15-2021': 18, '12-16-2021': 19,
            '12-18-2021': 20, '12-19-2021': 21, '12-19-2021b': 22, '01-16-2022': 23}

length = 200

for location in [2450, 175, 550, 575, 600, 650, 750, 825, 975, 1200, 1300, 1500, 1650, 1700, 2200, 2300, 2350, 2400,
                 2500, 2525]:
    # 指定要创建的文件夹路径
    # folder_path = "data_down/loc" + str(location) + "/input/"
    folder_lidar_path = "I:/Ithaca365/data/loc" + str(location) + "/lidar/"
    try:
        os.makedirs(folder_lidar_path)
        print("成功创建文件夹！")
    except FileExistsError:
        print("该文件夹已存在。")
    except Exception as e:
        print("发生错误：", str(e))

    # for date in ['11-23-2021', '11-29-2021', '12-02-2021', '12-03-2021', '12-08-2021', '12-09-2021', '12-13-2021', '12-18-2021', '12-19-2021', '12-19-2021b']:
    for date in ['11-19-2021', '11-22-2021', '11-30-2021', '12-01-2021', '12-06-2021', '12-07-2021', '12-14-2021',
                 '12-15-2021', '12-16-2021', '01-16-2022']:
        traversal_idx = date2num[date] - 1

        imgs0_path, imgs2_path, lidar_path = extract_path(location, traversal_idx, date, length)

        imgs0_sub_path = imgs0_path[::2]

        imgs2_sub_path = imgs2_path[::2]

        lidar_sub_path = lidar_path[::2]

        # for j in range(len(imgs0_sub_path)):
        #     # print(imgs0_sub_path[j].split('/'))
        #     shutil.copy(imgs0_sub_path[j], folder_path) 

        #     date_num = date.split('-')[0]+date.split('-')[1]
        #     loc_num = str(location).zfill(4)
        #     img_num = str(j).zfill(3)
        #     cam_num = '0'
        #     new_name =  date_num + loc_num + cam_num + img_num + '.png'    

        #     os.rename(folder_path + imgs0_sub_path[j].split('/')[-1], folder_path +new_name)

        # for j in range(len(imgs0_sub_path)):
        #     # print(imgs0_sub_path[j].split('/'))

        #     with Image.open(imgs0_sub_path[j]) as img:
        #         date_num = date.split('-')[0]+date.split('-')[1]
        #         loc_num = str(location).zfill(4)
        #         img_num = str(j).zfill(3)
        #         cam_num = '0'
        #         new_name =  date_num + loc_num + cam_num + img_num + '.png'  

        #         # Calculate the new size, half the original size
        #         new_size = (img.width // 2, img.height // 2)

        #         # Resize the image to the new size
        #         downsampled_img = img.resize(new_size)

        #         # Save the downsampled image to the specified output path
        #         downsampled_img.save(folder_path +new_name)

        #         # img.save(folder_path +new_name)

        '''
        for j in range(len(imgs2_sub_path)):
            # print(imgs2_sub_path[j].split('/'))
            shutil.copy(imgs2_sub_path[j], folder_path) 
            
            date_num = date.split('-')[0]+date.split('-')[1]
            loc_num = str(location).zfill(4)
            img_num = str(j).zfill(3)
            cam_num = '2'
            new_name =  date_num + loc_num + cam_num + img_num + '.png'    
            
            os.rename(folder_path + imgs2_sub_path[j].split('/')[-1], folder_path +new_name)
        '''

        for j in tqdm(range(len(imgs0_sub_path))):
            # print(imgs0_sub_path[j])
            shutil.copy(imgs0_sub_path[j], folder_lidar_path)

            date_num = date.split('-')[0] + date.split('-')[1]
            loc_num = str(location).zfill(4)
            img_num = str(j).zfill(3)
            cam_num = '2'
            new_name = date_num + loc_num + cam_num + img_num + '.bin'

            os.rename(folder_lidar_path + imgs0_sub_path[j].split('/')[-1], folder_lidar_path + new_name)
