import torch
import numpy as np
from diffusers import StableUnCLIPImg2ImgPipeline
import sys
import os
import time

import re

import pickle

import nibabel as nib
from nilearn.maskers import NiftiMasker
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def generate_image(pipe, embedding, image_name, diffusion_steps=21):
    '''
    Generate an image in Stable UnCLIP using a latent embedding.
        Inputs:
            pipe: StableUnCLIPImg2ImgPipeline object
            embedding: 1024-d numpy array
            image_name: output filename
            diffusion_steps: Number of diffusion steps for image generation. Can tweak this based on compute power.
        Outputs:
            Saved image.
    '''

    embedding = torch.tensor(np.reshape(
        embedding, (1, np.size(embedding))), dtype=torch.float16)
    # print(embedding.size())
    embedding = embedding.to('cuda')

    images = pipe(image_embeds=embedding, num_inference_steps=diffusion_steps).images
    images[0].save(image_name)


def prep_model(model_path):
    '''
    Helper function to load Stable UnCLIP model into memory.
    '''
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16).to('cuda')

    return pipe
    
    
    
def get_filtered_files(directory, start, end, pattern=r'^img_(\d{4}).nii.gz$'):
    '''
    Helper function to 'get_niftis()'. Get the aligned NifTI file names between the generation start time and its end time (plus a buffer).
        Inputs:
            directory: Aligned NifTI file directory.
            start: Start time in TRs relative to the first image in the run.
            end: End time in TRs.
            pattern: Regex pattern to find the relevant frames.
        Outputs:
            matching_files: A list of file names for the relevant frames.

    '''
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
    '''
    Function to get the aligned NifTI files within a specified time range in seconds, relative to the start of the run.
        Inputs:
            dir: Aligned NifTI file directory.
            start_time: Start time of the generation in seconds, relative to the beginning of the run.
            end_time: End time in seconds, plus a buffer.
            tr: Repetition time.
        Outputs:
            file_list: List of NifTI file paths for the relevant frames.
    '''

    start_tr = round(start_time / tr)
    end_tr = round(end_time / tr)
    duration = end_tr - start_tr

    file_list = get_filtered_files(dir, start_tr, end_tr)

    while len(file_list) < (duration):
        print(f"Not enough images found. Waiting for {duration - len(file_list)} more images...")
        time.sleep(1)
        file_list = get_filtered_files(dir, start_tr, end_tr)

    file_list = [os.path.join(dir, file) for file in file_list]

    return file_list


def create_4d_img(file_names):
    '''
    Function that takes a list of NifTI file names and returns a 4D image in nibabel format.
        Inputs:
            file_names: List of file paths to .nii.gz images.
        Outputs:
            new_img: 4D image in nibabel nifti1 format.
    '''

    # Load the first image to get the shape and affine
    first_image = nib.load(file_names[0])
    data_shape = first_image.shape
    affine = first_image.affine

    # Initialize an empty array with the shape for 4D data
    all_data = np.zeros((*data_shape, len(file_names)))

    # Load each 3D image and add it to the 4D array
    for i, file_name in enumerate(file_names):
        img = nib.load(file_name)
        all_data[..., i] = img.get_fdata()

    # Create a new NIfTI image
    new_img = nib.Nifti1Image(all_data, affine)

    return new_img

def get_scores(dir, onset_times, masker, pipeline, model):
    '''
    Evaluates a set of frames in order to get a list of predicted scores for the target.
        Inputs:
            dir: Aligned NifTI file directory.
            onset_times: List of onset times of images in one generation in seconds, relative to the start of the run.
            masker: NiftiMasker model to convert 4D data to a 2D matrix.
            pipeline: PCA + Scaler pipeline to convert masked data into reduced matrix.
            model: Logistic regression model for predicting scores from dimension-reduced data.
        Outputs:
            y_pred: Predicted scores on target measure.
    '''

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


    X = pipeline.transform(peak_frames)

    y_pred = model.predict_proba(X)[:,1]
    return y_pred


if __name__ == "__main__":

    '''
    This script combines:

    EVALUATOR
    This script is responsible for taking 1) the converted and aligned fMRI frames, and 2) the onset times of images in the psychopy program
    to compute scores on a target measure using pretrained models (logistic regression) and pipelines (masking, PCA, and scaling).

    It runs asycnhronously to process frames as soon as a generation is complete.

    GENERATOR
    It also waits for the psychopy program to read the scores and output the next generation's embeddings,
    which it uses to produce a new set of images.
    '''

    model_path = r"D:\TDLab\pretrained_models\stable-diffusion-2-1-unclip"
    if not os.path.exists(model_path):
        print("Primary model path not found. Using secondary (offline) model path...")
        model_path = "../../pretrained_models/stable-diffusion-2-1-unclip"

    pipe = prep_model(model_path)

    shared_drive_path = sys.argv[1]
    root_dir = sys.argv[2]
    participant = int(sys.argv[3])
    target = sys.argv[4]

    embeddings_files_complete = []
    
    onset_files_complete = []
    
    nifti_dir = os.path.join(shared_drive_path, "data", "aligned")


    with open(f"{shared_drive_path}/models/sub-{participant:02}/masker_model.pkl", "rb") as f:
        masker = pickle.load(f)

    with open(f"{shared_drive_path}/models/sub-{participant:02}/pipeline_model.pkl", "rb") as f:
        pipeline = pickle.load(f)

    with open(f"{shared_drive_path}/models/sub-{participant:02}/{target}_model.pkl", "rb") as f:
        model = pickle.load(f)

    while(True):
    
        ## Generator loop

        embeddings_files = []
        embeddings_paths = []

        # Search for new embeddings files
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename == "embeddings.txt":
                    full_path = os.path.join(dirpath, filename)
                    embeddings_paths.append(dirpath)
                    embeddings_files.append(full_path)

        difference_list = [file for file in embeddings_files if file not in embeddings_files_complete]

        if len(difference_list) == 1:

            output_path = embeddings_paths[-1]
            
            time.sleep(1)

            embeddings = np.loadtxt(difference_list[0], delimiter=',')
            finished = np.zeros(embeddings.shape[0])

            for i in range(embeddings.shape[0]):

                filename = os.path.join(output_path, f"img_{i:02}.png")
                generate_image(pipe, embeddings[i,:], filename, diffusion_steps=5) ## Edit diffusion steps based on performance
                finished[i] = 1
                np.savetxt(f"{output_path}/status.txt", finished, delimiter=',')

            embeddings_files_complete.append(difference_list[0])

        elif len(difference_list) > 1:
            raise Exception("Too many embeddings files found!")
        else:
            pass
            
            
        ## Evaluator loop
            
        onset_files = []
        onset_paths = []

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

            fitness = get_scores(nifti_dir, onset_times, masker, pipeline, model)
            np.savetxt(f"{output_path}/fitness.txt", fitness, delimiter=',')

            onset_files_complete.append(difference_list[0])
            
            if len(onset_files_complete) == 10:
                print("Run complete.")
                exit()

        elif len(difference_list) > 1:
            raise Exception("Too many onset files found!")
        else:
            pass

        time.sleep(1)




