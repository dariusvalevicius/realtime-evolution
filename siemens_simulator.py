
import shutil
import os
import time
import sys


if __name__ == "__main__":

    shared_drive_path = sys.argv[1]

    source_path = f"{shared_drive_path}/data/pre_source/"
    dest_path = f"{shared_drive_path}/data/source/"

    # anat_transforms = os.listdir


    tr = 0.867

    file_list = os.listdir(source_path)
    # print(file_list)

    for folder in ["source", "dicom", "nifti", "aligned"]:
        for root, dirs, files in os.walk(f"{shared_drive_path}/data/{folder}"):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))



    for i, file in enumerate(file_list):

        name, ext = os.path.splitext(file)

        time.sleep(tr)

        # dest_folder = os.path.join(dest_path, f"dcm_{i:04}")
        # os.mkdir(dest_folder)
        shutil.copy(os.path.join(source_path, file), dest_path)

