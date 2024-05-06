import numpy as np
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import csv
import pandas as pd

def extract_number(file_path):
    fname = Path(file_path).name  # 直接从文件路径获取文件名
    numbers = ''.join(filter(str.isdigit, fname))  # 提取文件名中的所有数字
    return int(numbers) if numbers else -1  # 将数字部分转换为整数

# 定义排序函数
def sort_key(file_path):
    filename = Path(file_path).name
    priority_sequence = ['0116', '1119', '1122', '1130', '1201', '1206', '1207', '1214', '1215', '1216']
    # 提取文件名中的数字部分
    parts = ''.join(filter(str.isdigit, filename))
    prefix, suffix = parts[:4], parts[-2:]  # 分别提取前四位和后两位数字

    # 获取前四位数字的优先级索引，如果不存在于列表中，则返回一个较大的数字保证它排在最后
    prefix_priority = priority_sequence.index(prefix) if prefix in priority_sequence else len(priority_sequence)

    return (prefix_priority, int(suffix))  # 返回一个元组，包含前四位的优先级和后两位的数字


if __name__ == '__main__':

    file = open("depth_eval_875.csv", "w")
    file = csv.writer(file)
    file.writerow(["", "AbsRel", "RMSE", "SqRel", "RMSElog", "Delta1", "Delta2", "Delta3"])
    root_dir = "I:/Ithaca365"
    for loc in ["loc175", "loc550", "loc575", "loc600", "loc650", "loc750", "loc825", "loc975", "loc1200", "loc1500",
                    "loc1650", "loc1700", "loc2200", "loc2300", "loc2350", "loc2400", "loc2450", "loc2500", "loc2525"]:
        # if loc != "loc1200":
        #     continue

        # rendered_depth_dir = f'{root_dir}/data/{loc}/output/depth_sky_original_unit'
        # gt_depth_dir = f'{root_dir}/data/{loc}/output/gt_depth_sky_original_unit'
        rendered_depth_dir = f'{root_dir}/cam2_data/depth_sky_original_unit_1000/{loc}'
        gt_depth_dir = f'{root_dir}/cam2_data/gt_depth_original_unit_1000/{loc}'

        # 加载和排序rendered depth文件
        # rendered_files = sorted(os.listdir(rendered_depth_dir), key=lambda x: extract_number(os.path.join(rendered_depth_dir, x)))
        rendered_files = os.listdir(rendered_depth_dir)
        rendered_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        # 加载和排序gt depth文件
        # gt_files = sorted(os.listdir(gt_depth_dir), key=lambda x: sort_key(os.path.join(gt_depth_dir, x)))
        gt_files = os.listdir(gt_depth_dir)
        gt_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        # 确认文件列表不为空
        assert rendered_files and gt_files, "File lists are empty!"

        total_AbsRel = 0
        total_RMSE = 0
        total_RMSE_log = 0
        total_SqRel = 0
        count = 0
        delta_125 = 0
        delta_125_2 = 0
        delta_125_3 = 0

        counter = 0

        # 演示如何加载对应的文件（例如：计算MSE）
        for rendered_file, gt_file in zip(rendered_files, gt_files):  # 注意每隔8个取一个rendered depth与gt depth对应
            if not counter % 8 == 0:
                rendered_depth = np.load(os.path.join(rendered_depth_dir, rendered_file))
                gt_depth = np.load(os.path.join(gt_depth_dir, gt_file))

                gt_depth_map = torch.tensor(gt_depth)
                rendered_depth = torch.tensor(rendered_depth).squeeze(0)

                gt_depth_map = gt_depth_map.unsqueeze(0).unsqueeze(0)
                # gt_depth_map = F.interpolate(gt_depth_map, size=(604, 960), mode='bilinear',
                #                              align_corners=True)
                gt_depth_map = F.interpolate(gt_depth_map, size=(1208, 1920), mode='bilinear',
                                             align_corners=True)
                gt_depth_map = gt_depth_map.squeeze(0).squeeze(0)

                # 这里可以添加评估代码，例如计算MSE等
                # print(f"Loaded {rendered_file} and {gt_file}")

                valid_mask = gt_depth_map > 0
                gt_depth_map = gt_depth_map[valid_mask]
                rendered_depth = rendered_depth[valid_mask]

                # 计算指标
                abs_diff = torch.abs(gt_depth_map - rendered_depth)
                abs_rel = torch.mean(abs_diff / gt_depth_map)
                # print("GroundTr", gt_depth_map)
                # print("Rendered", rendered_depth)
                # print("Abs_Diff", abs_diff)
                # print("AbsRatio", abs_diff / gt_depth_map)
                # print(abs_rel)
                # print("-----------------")
                sq_rel = torch.mean(((gt_depth_map - rendered_depth) ** 2) / gt_depth_map)

                rmse = torch.sqrt(torch.mean((gt_depth_map - rendered_depth) ** 2))
                rmse_log = torch.sqrt(torch.mean((torch.log(gt_depth_map + 1) - torch.log(rendered_depth + 1)) ** 2))

                n_valid_element = float(gt_depth_map.size(0))
                y_over_z = torch.div(gt_depth_map, rendered_depth)
                z_over_y = torch.div(rendered_depth, gt_depth_map)
                max_ratio = torch.max(y_over_z, z_over_y)
                # error['D_DELTA1.02'] = torch.div(torch.sum(max_ratio <= 1.02), n_valid_element)
                # error['D_DELTA1.05'] = torch.div(torch.sum(max_ratio <= 1.05), n_valid_element)
                # error['D_DELTA1.10'] = torch.div(torch.sum(max_ratio <= 1.10), n_valid_element)
                delta_125 += torch.div(torch.sum(max_ratio <= 1.25), n_valid_element).numpy()
                delta_125_2 += torch.div(torch.sum(max_ratio <= 1.25 ** 2), n_valid_element).numpy()
                delta_125_3 += torch.div(torch.sum(max_ratio <= 1.25 ** 3), n_valid_element).numpy()

                # 累加计算总值
                total_AbsRel += abs_rel.item()
                total_RMSE += rmse.item()
                total_RMSE_log += rmse_log.item()
                total_SqRel += sq_rel.item()
                count += 1
            counter += 1

        # 计算平均值
        mean_AbsRel = total_AbsRel / count
        mean_RMSE = total_RMSE / count
        mean_RMSE_log = total_RMSE_log / count
        mean_SqRel = total_SqRel / count
        delta_125 = delta_125 / count
        delta_125_2 = delta_125_2 / count
        delta_125_3 = delta_125_3 / count

        file.writerow([loc, mean_AbsRel, mean_RMSE, mean_SqRel, mean_RMSE_log, delta_125, delta_125_2, delta_125_3])

        print(loc)
        print(f"Mean AbsRel: {mean_AbsRel}")
        print(f"Mean RMSE: {mean_RMSE}")
        print(f"Mean RMSE (log): {mean_RMSE_log}")
        print(f"Mean SqRel: {mean_SqRel}")
        print(f"delta_125: {delta_125}")
        print(f"delta_125_2: {delta_125_2}")
        print(f"delta_125_3: {delta_125_3}")
        print("")

    df = pd.read_csv("depth_eval_875.csv")
    GFG = pd.ExcelWriter('depth_eval_875.xlsx')
    df.to_excel(GFG, index=False, float_format='%11.3f')

    GFG.close()






# if __name__ == '__main__':
#     # rendered_depth_dir = 'I:/Ithaca365/loc2450/output/loc2450_segformer_idso_2/test/ours_30000/depth_map'
#     # rendered_depth_dir = 'I:/Ithaca365/loc2450/rendered_depth_original_unit'
#     gt_depth_dir = 'I:/Ithaca365/loc2450/gt_depth_masked_original_unit'
#
#     rendered_depth_dir = f'I:/Ithaca365/depth_sky_original_unit/loc2450'
#     # rendered_depth_dir = f'I:/Ithaca365/data/loc2450/output/depth_sky_original_unit'
#     # gt_depth_dir = f'I:/Ithaca365/data/loc2450/output/gt_depth_sky_original_unit'
#     # gt_depth_dir = f'I:/Ithaca365/gt_depth_sky_original_unit/loc2450'
#
#     # rendered_depth_dir = 'I:/Ithaca365/loc2450/depth_anything_original_unit'
#     # gt_depth_dir = 'I:/Ithaca365/loc2450/gt_depth_original_unit'
#
#     # rendered_depth_dir = f'{root_dir}/data/{loc}/output/depth_sky_original_unit'
#     # gt_depth_dir = f'{root_dir}/data/{loc}/output/gt_depth_sky_original_unit'
#
#     # 加载和排序rendered depth文件
#     # rendered_files = sorted(os.listdir(rendered_depth_dir), key=lambda x: extract_number(os.path.join(rendered_depth_dir, x)))
#     rendered_files = os.listdir(rendered_depth_dir)
#     rendered_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
#     # 加载和排序gt depth文件
#     # gt_files = sorted(os.listdir(gt_depth_dir), key=lambda x: sort_key(os.path.join(gt_depth_dir, x)))
#     gt_files = os.listdir(gt_depth_dir)
#     gt_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
#
#     # 确认文件列表不为空
#     assert rendered_files and gt_files, "File lists are empty!"
#
#     total_AbsRel = 0
#     total_RMSE = 0
#     total_RMSE_log = 0
#     total_SqRel = 0
#     count = 0
#     delta_125 = 0
#     delta_125_2 = 0
#     delta_125_3 = 0
#
#     # 演示如何加载对应的文件（例如：计算MSE）
#     for rendered_file, gt_file in zip(rendered_files, gt_files):  # 注意每隔8个取一个rendered depth与gt depth对应
#         rendered_depth = np.load(os.path.join(rendered_depth_dir, rendered_file))
#         gt_depth = np.load(os.path.join(gt_depth_dir, gt_file))
#
#         gt_depth_map = torch.tensor(gt_depth)
#         rendered_depth = torch.tensor(rendered_depth).squeeze(0)
#
#         gt_depth_map = gt_depth_map.unsqueeze(0).unsqueeze(0)
#         # gt_depth_map = F.interpolate(gt_depth_map, size=(604, 960), mode='bilinear',
#         #                              align_corners=True)
#         gt_depth_map = F.interpolate(gt_depth_map, size=(1208, 1920), mode='bilinear',
#                                      align_corners=True)
#         gt_depth_map = gt_depth_map.squeeze(0).squeeze(0)
#
#         # 这里可以添加评估代码，例如计算MSE等
#         # print(f"Loaded {rendered_file} and {gt_file}")
#
#         valid_mask = gt_depth_map > 0
#         gt_depth_map = gt_depth_map[valid_mask]
#         rendered_depth = rendered_depth[valid_mask]
#
#         # 计算指标
#         abs_diff = torch.abs(gt_depth_map - rendered_depth)
#         abs_rel = torch.mean(abs_diff / gt_depth_map)
#         sq_rel = torch.mean(((gt_depth_map - rendered_depth) ** 2) / gt_depth_map)
#
#         # if loc == "loc1500":
#         #     print()
#         #     print(gt_depth_map - rendered_depth)
#         rmse = torch.sqrt(torch.mean((gt_depth_map - rendered_depth) ** 2))
#         rmse_log = torch.sqrt(torch.mean((torch.log(gt_depth_map + 1) - torch.log(rendered_depth + 1)) ** 2))
#
#         n_valid_element = float(gt_depth_map.size(0))
#         y_over_z = torch.div(gt_depth_map, rendered_depth)
#         z_over_y = torch.div(rendered_depth, gt_depth_map)
#         max_ratio = torch.max(y_over_z, z_over_y)
#         # error['D_DELTA1.02'] = torch.div(torch.sum(max_ratio <= 1.02), n_valid_element)
#         # error['D_DELTA1.05'] = torch.div(torch.sum(max_ratio <= 1.05), n_valid_element)
#         # error['D_DELTA1.10'] = torch.div(torch.sum(max_ratio <= 1.10), n_valid_element)
#         delta_125 += torch.div(torch.sum(max_ratio <= 1.25), n_valid_element).numpy()
#         delta_125_2 += torch.div(torch.sum(max_ratio <= 1.25 ** 2), n_valid_element).numpy()
#         delta_125_3 += torch.div(torch.sum(max_ratio <= 1.25 ** 3), n_valid_element).numpy()
#
#         # 累加计算总值
#         total_AbsRel += abs_rel.item()
#         total_RMSE += rmse.item()
#         total_RMSE_log += rmse_log.item()
#         total_SqRel += sq_rel.item()
#         count += 1
#
#     # 计算平均值
#     mean_AbsRel = total_AbsRel / count
#     mean_RMSE = total_RMSE / count
#     mean_RMSE_log = total_RMSE_log / count
#     mean_SqRel = total_SqRel / count
#     delta_125 = delta_125 / count
#     delta_125_2 = delta_125_2 / count
#     delta_125_3 = delta_125_3 / count
#
#     print(f"Mean AbsRel: {mean_AbsRel}")
#     print(f"Mean RMSE: {mean_RMSE}")
#     print(f"Mean RMSE (log): {mean_RMSE_log}")
#     print(f"Mean SqRel: {mean_SqRel}")
#     print(f"delta_125: {delta_125}")
#     print(f"delta_125_2: {delta_125_2}")
#     print(f"delta_125_3: {delta_125_3}")
#     print("")
