# Realtime Evolution of AI Images using fMRI

Under construction! This project is in the pilot phase, and may not work out-of-the-box on a new machine with new data.

This repo contains code for running real-time image evolution with fMRI. The program is recommended to be run on two computers: One for running the psychopy program and presenting the stimuli, and one for doing the image processing, model predictions, and AI image generation. The secondary (support) computer will need a sufficiently powerful NVIDIA GPU to be able to do the image generation at a rate quick enough for real-time.

Most new gaming desktops should be able to produce an image every 2-8 seconds; the number of diffusion steps can be tweaked to account for differences in computing power.

## Requirements

This pipeline require an installation of pytorch with CUDA capability on the support computer. This is not included in the requirements.txt, as the version must be compatible with the machine's hardware and drivers.

Python 3.8.x (3.8.10 tested) is required for stable use of psychopy.

An installation of Stable unCLIP is required, which can be downloaded from HuggingFace. The regular (1024 dim) model is used by default, but the pipeline can be tweaked to use the small (768 dim) model for lower compute requirements.

## Psychopy program (image_gen_mri.py)

This program presents images to the participant and recomputes new latent space embeddings based on the scores provided by the evaluator script. It operates in a serial flow with the various components of the support computer scripts.

## Support computer programs

### manager.py

This script simply starts the various components of the support computer pipeline as subprocesses. Running multiple processes in parallel is necessary to avoid bottlenecks, such as waiting for a full set of images to generate after new embeddings are computed. Instead, the scripts can listen for their respective inputs and start computations as soon as the minimal amount are received.

### processor.py

This script implements data conversion and alignment/motion correction, similarly to the sister repo for decoder construction. In fact, the alignment code is identical to the decoder construction alignment, in order to ensure comparable results between offline and online modeling.

### collector.py (combines generator.py and evaluator.py)

This script listens for onset time files in the output directory, indicating that a generation presentation has been completed by the psychopy program. After one full generation, the evaluator code can use the saved (pickled) model files to create fear predictions for every image. These values are saved and passed to the psychopy program for recombination.

Once a new embeddings file is created for the next generation, the generator code produces the next set of images. The psychopy program waits for the first image and begins presenting the new set immediately. The number of diffusion steps should be tweaked so that the generation time approxiately matches the desired stimulus presentation time, which by default is 4 seconds (3s duration + 1s washout).