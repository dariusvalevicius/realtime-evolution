import os, sys
import subprocess
import time
from glob import glob

import ants

import numpy as np
import pandas as pd
from scipy.io import loadmat

def load_matrix_in_any_format(filepath):
    _, ext = os.path.splitext(filepath)
    if ext == '.txt':
        data = np.loadtxt(filepath)
    elif ext == '.npy':
        data = np.load(filepath)
    elif ext == '.mat':
        # .mat are actually dictionnary. This function support .mat from
        # antsRegistration that encode a 4x4 transformation matrix.
        transfo_dict = loadmat(filepath)
        lps2ras = np.diag([-1, -1, 1])

        rot = transfo_dict['AffineTransform_float_3_3'][0:9].reshape((3, 3))
        trans = transfo_dict['AffineTransform_float_3_3'][9:12]
        offset = transfo_dict['fixed']
        r_trans = (np.dot(rot, offset) - offset - trans).T * [1, 1, -1]

        data = np.eye(4)
        data[0:3, 3] = r_trans
        data[:3, :3] = np.dot(np.dot(lps2ras, rot), lps2ras)
    else:
        raise ValueError('Extension {} is not supported'.format(ext))

    return data

def affine_to_regressors(affine):
    motion_params = np.zeros(6)

    # Extract translations
    t_x, t_y, t_z = affine[:3, 3]
    motion_params[:3] = [t_x, t_y, t_z]

    # Extract rotations (assume small angles)
    R = affine[:3, :3]
    pitch = np.arctan2(R[2, 1], R[2, 2])  # Rotation around X-axis
    yaw = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))  # Rotation around Y-axis
    roll = np.arctan2(R[1, 0], R[0, 0])  # Rotation around Z-axis
    motion_params[3:] = [pitch, yaw, roll]
    # print(motion_params)
    return motion_params


if __name__ == "__main__":

    '''
    This script performs several functions:
        1) Image conversion (DCM -> NifTI)
        2) Image alignment using antspyx

    It runs asychronously to process DCM frames as soon as they are written to the shared drive.
    '''

    print("PROCESSOR is starting...")

    shared_drive_path = sys.argv[1]
    participant = int(sys.argv[2])

    # Set alignment target
    ref_img = ants.image_read(f"{shared_drive_path}/models/sub-{participant:02}/ref_img.nii.gz")

    # Watch output directory for dicom files
    source_dir = glob(f"{shared_drive_path}/data/source/*realtime*")
    while not source_dir:
        time.sleep(1)
        source_dir = glob(f"{shared_drive_path}/data/source/*realtime*")
        continue
        
    source_dir = source_dir[0]
    
    print(f"PROCESSOR: Processing: {source_dir}")

    nifti_dir = os.path.join(shared_drive_path, "data", "nifti")
    aligned_dir = os.path.join(shared_drive_path, "data", "aligned")

    processed_dcms = []
    unprocessed_dcms = []

    t1 = time.time()
    checkpoint = 10

    # Create motion params df
    motion_param_names = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    motion_df = pd.DataFrame(columns = motion_param_names)
    motion_df_path = os.path.join(aligned_dir, "motion_params.tsv")
    motion_df.to_csv(motion_df_path, index=False, sep="\t")


    while(True):

        dicoms = []

        for file in os.listdir(source_dir):
            dicoms.append(file)


        s = set(processed_dcms)
        unprocessed_dcms = [file for file in dicoms if file not in s]

        if (len(unprocessed_dcms) >= 1):
            # If there is no file queue, wait a bit to ensure file is fully written
            if (len(unprocessed_dcms) == 1):
                time.sleep(2)
            else:
                pass
            
            for file in unprocessed_dcms:
                
                dicom_file = os.path.join(source_dir, file)
                output_filename = f"img_{len(processed_dcms):04}"

                # Construct the dcm2niix command
                command = [
                    'dcm2niix', 
                    '-o', nifti_dir,    # Output directory
                    '-f', output_filename,  # Output filename
                    '-b', 'n',
                    '-v', '0',
                    '-s', 'y', # Single file mode
                    dicom_file            # Input DICOM file
                ]

                # Run the command using subprocess
                dcm2nii = subprocess.call(command, stdout=open(os.devnull, 'wb'))

                # Do registration
                nifti_file = os.path.join(nifti_dir, output_filename + '.nii')
                ants_img = ants.image_read(nifti_file)

                # areg = align_image(ants_img, fmri_ref, target_img, transforms)
                areg = ants.registration( ref_img, ants_img, "BOLDAffine" )
                save_filename = os.path.join(aligned_dir, output_filename + ".nii.gz")
                ants.image_write(areg['warpedmovout'], save_filename)

                # Extract transformation matrix (motion parameters)
                transform = areg['fwdtransforms'][0]  # Path to the transform file
                affine = load_matrix_in_any_format(transform)
                motion_params = affine_to_regressors(affine)
                motion_df = pd.DataFrame(data = motion_params.reshape(1,-1), columns = motion_param_names)
                motion_df.to_csv(motion_df_path, mode='a', index=False, header=False, sep="\t")

                processed_dcms.append(file)

        aligned_imgs = os.listdir(aligned_dir)

        # Print status intermittently
        t2 = time.time()
        if int(t2 - t1) >= checkpoint:

            print(f"PROCESSOR: Processed {len(aligned_imgs)} images in {int(t2-t1)} seconds, at a rate of {len(aligned_imgs) / int(t2-t1)} images per second.")

            checkpoint = checkpoint + 10