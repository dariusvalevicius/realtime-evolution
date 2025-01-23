
import shutil
import os
import time
import sys
from glob import glob


if __name__ == "__main__":
    '''
    This script simulates an MRI scanner by copying DCM files from a source directory
    to the 'dicom' folder where they will be read from by the realtime program.
    '''
    print("SIMULATOR is starting...")

    shared_drive_path = sys.argv[1]

    source_path = f"{shared_drive_path}/data/simulation"
    dest_path = f"{shared_drive_path}/data/source/realtime_simulated"
    os.mkdir(dest_path)


    tr = 0.867

    file_list = os.listdir(source_path)


    for i, file in enumerate(file_list):

        name, ext = os.path.splitext(file)

        time.sleep(tr)

        shutil.copy(os.path.join(source_path, file), dest_path)


