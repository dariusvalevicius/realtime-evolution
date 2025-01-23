import torch
import numpy as np
import pandas as pd

from diffusers import StableUnCLIPImg2ImgPipeline
import sys
import os
import time
from PIL import Image

import re

import pickle
import json

import nibabel as nib
from nilearn.maskers import NiftiMasker
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline




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

def get_embed(path, vision_model, processor):
    '''
    Gets the CLIP embeddings from an image file.
    '''
    
    image = Image.open(path)
    inputs = processor(text=None, images=image, return_tensors="pt")
    # print(type(inputs['pixel_values']))

    pixel_values = inputs['pixel_values'].to('cuda')

    outputs = vision_model(pixel_values)

    image_features = outputs.image_embeds
    image_embeddings = torch.Tensor.cpu(image_features).detach().numpy()[0, :]
    # print(image_features.size())

    return image_embeddings

    
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
        time.sleep(2)
        file_list = get_filtered_files(dir, start_tr, end_tr)

    file_list = [os.path.join(dir, file) for file in file_list]

    idx = np.arange(start_tr, end_tr + 1)

    return file_list, idx


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

def get_scores(dir, onset_times, masker, target):
    '''
    Evaluates a set of frames in order to get a list of predicted scores for the target.
        Inputs:
            dir: Aligned NifTI file directory.
            onset_times: List of onset times of images in one generation in seconds, relative to the start of the run.
            masker: NiftiMasker model to convert 4D data to a 2D matrix.
            target: Weight vector to be used for Pearson-type decoding.
        Outputs:
            y_pred: Predicted scores on target measure.
    '''

    # Get nifti image for onset - 5s to last onset + 10s
    start_time = onset_times[0]
    end_time = onset_times[-1] + 8

    list, idx = get_niftis(dir, start_time, end_time)

    img = create_4d_img(list)
    print(f"Image size: {img.shape}")

    # Get motion params for cleaning
    motion_params = pd.read_csv(os.path.join(dir, "motion_params.tsv"), sep="\t").iloc[idx, :]

    # Mask image
    masked_img = masker.transform(img, confounds=motion_params)

    # Get peak frames
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

    y_pred = []

    for i in range(peak_frames.shape[0]):
        r_val = np.corrcoef(peak_frames[i,:], target)[0,1]
        y_pred.append(r_val)

    y_pred = np.array(y_pred)

    return y_pred, peak_frames


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

    print("COLLECTOR is starting...")

    # Get arguments
    shared_drive_path = sys.argv[1]
    root_dir = sys.argv[2]
    participant = int(sys.argv[3])
    target = sys.argv[4]
    condition = sys.argv[5]

    with open('config.json') as f:
        config = json.load(f)

    model_path = config["unclip_model_path"]
    diffusion_steps = config["diffusion_steps"]
    max_iters = config["max_iters"]
    smoothing_fwhm = config["smoothing_fwhm"]
    t_r = config["t_r"]

    # Prepare generator model
    pipe = prep_model(model_path)
    vision_model = pipe.image_encoder
    processor = pipe.feature_extractor

    embeddings_files_complete = []
    onset_files_complete = []
    nifti_dir = os.path.join(shared_drive_path, "data", "aligned")

    if condition == "brain":
        # Load fMRI masker and target model
        mask_img = nib.load(f"{shared_drive_path}/models/sub-{participant:02}/mask_img.nii.gz")
        target_img = nib.load(f"{shared_drive_path}/models/sub-{participant:02}/{target}_img.nii.gz")
        ref_img = nib.load(f"{shared_drive_path}/models/sub-{participant:02}/ref_img.nii.gz")

        # Initialize masker
        masker = NiftiMasker(
            mask_img=mask_img,
            t_r=t_r,
            smoothing_fwhm=smoothing_fwhm,
            standardize=True,
            detrend=True
        )

        masker.fit(ref_img)
        target_img_flat = masker.transform(target_img)


    ## Start loop
    for iter in range(max_iters):

        # Wait for folder to appear
        gen_folder = os.path.join(root_dir, f"generation_{iter:02}")
        while not os.path.exists(gen_folder):
            time.sleep(1)

        
        # Wait for embedding file to appear
        embeddings_path = os.path.join(gen_folder, "embeddings.txt")
        while not os.path.isfile(embeddings_path):
            time.sleep(1)

        # Check if file is written before loading
        while True:
            try:
                with open(embeddings_path, 'rb') as _:
                    _.close()
                    break
            except IOError:
                time.sleep(1)
        
        embeddings = np.loadtxt(embeddings_path, delimiter=',')

        embeddings_post = np.empty_like(embeddings)
        finished = np.zeros(embeddings.shape[0])

        for i in range(embeddings.shape[0]):

            filename = os.path.join(gen_folder, f"img_{i:02}.png")
            # Generate image
            generate_image(pipe, embeddings[i,:], filename, diffusion_steps=diffusion_steps) ## Edit diffusion steps based on performance
            # Read back embeddings
            this_embedding_post = get_embed(filename, vision_model, processor)
            embeddings_post[i,:] = this_embedding_post

            # Set status to complete
            finished[i] = 1
            np.savetxt(os.path.join(gen_folder, "status.txt"), finished, delimiter=',')

        # Save post embeddings
        np.savetxt(os.path.join(gen_folder, "embeddings_post.txt"), embeddings_post, delimiter=',')
            
        # Wait for onsets file
        onsets_path = os.path.join(gen_folder, "onset_times.txt")
        while not os.path.isfile(onsets_path):
            time.sleep(1)

        # Check if file is written before loading
        while True:
            try:
                with open(onsets_path, 'rb') as _:
                    _.close()
                    break
            except IOError:
                time.sleep(1)
        
        onset_times = np.loadtxt(onsets_path, delimiter=',')

        if condition == "brain":
            # Compute and save fitness scores
            fitness, peak_frames = get_scores(nifti_dir, onset_times, masker=masker, target=target_img_flat)
            np.savetxt(os.path.join(gen_folder, "fitness.txt"), fitness, delimiter=',')
            # Save computed trial responses
            np.savetxt(os.path.join(gen_folder, "peak_frames.txt"), peak_frames, fmt='%.4f', delimiter=",")

        else:
            print("Condition is: ratings. No processing needed this time.")

        print(f"COLLECTOR: Generation {iter} processing complete. Moving on...")


print("COLLECTOR finished processing. Closing...")


