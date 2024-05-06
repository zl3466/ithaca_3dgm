import os

import numpy as np
import cv2
import osgeo_utils.auxiliary.util

if __name__ == '__main__':
    root_dir = "I:/Ithaca365/loc2450"
    mask_dir = f"{root_dir}/seg_mask"
    mask_list = os.listdir(mask_dir)
    for i in range(len(mask_list)):
        mask_filename = mask_list[i]
        mask = np.load(f"{mask_dir}/{mask_filename}")
        print(mask.shape)
        print(np.where(mask != 1))
        out_dir = f"{mask_dir}/../mask_viz"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        cv2.imwrite(f"{out_dir}/{mask_filename.split('.')[0]}.png", mask*256)
