
import os, sys
import re
import time

import numpy as np

import pickle

import nibabel as nib
from nilearn.maskers import NiftiMasker
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression



import matplotlib.pyplot as plt


def get_filtered_files(directory, start, end, pattern=r'^img_(\d{4}).nii.gz$'):
    # Define the regex pattern for matching files
    pattern = re.compile(pattern)
    
    # List to store matching files
    matching_files = []
    
    # Iterate through the files in the directory
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            # Extract the number from the filename
            num = int(match.group(1))
            # Check if the number is within the specified range
            if start <= num <= end:
                matching_files.append(filename)
    
    return matching_files


def get_niftis(dir, start_time, end_time, tr=0.867):

    start_tr = round(start_time / tr)
    end_tr = round(end_time / tr)
    duration = end_tr - start_tr

    # file_dir = os.path.join("data", "aligned")

    file_list = get_filtered_files(dir, start_tr, end_tr)

    while len(file_list) < (duration):
        print(f"Not enough images found. Waiting for {duration - len(file_list)} more images...")
        time.sleep(1)
        file_list = get_filtered_files(dir, start_tr, end_tr)


    return [os.path.join(dir, file) for file in file_list]


def create_4d_img(file_names):
    # Load the first image to get the shape and affine
    first_image = nib.load(file_names[0])
    data_shape = first_image.shape
    affine = first_image.affine

    # Initialize an empty array with the shape for 4D data
    # (x, y, z, time points)
    all_data = np.zeros((*data_shape, len(file_names)))
    # print(all_data.shape)

    # Load each 3D image and add it to the 4D array
    for i, file_name in enumerate(file_names):
        img = nib.load(file_name)
        all_data[..., i] = img.get_fdata()

    # Create a new NIfTI image
    new_img = nib.Nifti1Image(all_data, affine)

    return new_img

def get_scores(dir, onset_times, masker, pca, model):

    # Get nifti image for onset - 5s to last onset + 10s
    start_time = onset_times[0]
    end_time = onset_times[-1] + 10

    list = get_niftis(dir, start_time, end_time)

    img = create_4d_img(list)
    print(f"Image size: {img.shape}")

    masked_img = masker.fit_transform(img)

    peak_frames = np.zeros((len(onset_times), masked_img.shape[1]))
    tr = 0.867

    for j, onset_time in enumerate(onset_times):

        # onset_in_tr = round(onset_time / tr)
        onset_in_tr = round(onset_times[0] / tr)

        peak_in_tr = round((onset_time + 6) / tr) - onset_in_tr

        fmri_avg = np.mean(masked_img[peak_in_tr - 1:peak_in_tr+2,:], axis=0)
        nan_indices = np.isnan(fmri_avg)
        if nan_indices.any():
            print(f"NaNs found in row: {j}")
        peak_frames[j,:] = fmri_avg


    X = pca.transform(peak_frames)

    y_pred = model.predict_proba(X)[:,1]
    return y_pred


if __name__ == "__main__":
    
    shared_drive_path = sys.argv[1]
    root_dir = sys.argv[2]
    participant = int(sys.argv[3])

    nifti_dir = os.path.join(shared_drive_path, "data", "aligned")


    onset_files_complete = []

    with open(f"{shared_drive_path}/models/sub-{participant:02}/masker_model.pkl", "rb") as f:
        masker = pickle.load(f)

    with open(f"{shared_drive_path}/models/sub-{participant:02}/pca_model.pkl", "rb") as f:
        pca = pickle.load(f)

    with open(f"{shared_drive_path}/models/sub-{participant:02}/fear_model.pkl", "rb") as f:
        model = pickle.load(f)


    while(True):


        onset_files = []
        onset_paths = []

        # root_dir = f"images/sub-{1:02}/run-{1}"
        # participant = 1

        # Search for new embeddings files
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename == "onset_times.txt":
                    full_path = os.path.join(dirpath, filename)
                    onset_paths.append(dirpath)
                    onset_files.append(full_path)

        difference_list = [file for file in onset_files if file not in onset_files_complete]

        if len(difference_list) == 1:

            print(f"Processing file: {difference_list[0]}")
            output_path = onset_paths[-1]
            time.sleep(1)

            onset_times = np.loadtxt(difference_list[0], delimiter=',')
            print(onset_times)

            fitness = get_scores(nifti_dir, onset_times, masker, pca, model)
            np.savetxt(f"{output_path}/fitness.txt", fitness, delimiter=',')

            onset_files_complete.append(difference_list[0])

        elif len(difference_list) > 1:
            raise Exception("Too many onset files found!")
        else:
            pass

        time.sleep(1)



    



