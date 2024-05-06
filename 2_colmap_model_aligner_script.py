import subprocess

def run_command_with_loc(loc):
    command_template = f".\\COLMAP.bat model_aligner --input_path I:/Ithaca365/data/{loc}/sparse/0 --output_path I:/Ithaca365/data/{loc}/colmap_cam_pose --ref_images_path I:/Ithaca365/data/{loc}/colmap_cam_pose/ref_pose.txt --ref_is_gps 0 --alignment_type custom --alignment_max_error 3.0 --transform_path I:/Ithaca365/data/{loc}/colmap_cam_pose/transform.txt"
    command = command_template.format(loc=loc)
    subprocess.run(command, shell=True)

if __name__ == '__main__':
    root_dir = f"I:/Ithaca365"
    for loc in ["loc175", "loc550", "loc575", "loc600", "loc650", "loc750", "loc825", "loc975", "loc1200", "loc1500",
                "loc1650", "loc1700", "loc2200", "loc2300", "loc2350", "loc2400", "loc2450", "loc2500", "loc2525"]:
        run_command_with_loc(loc)