
import torch
import numpy as np
from diffusers import StableUnCLIPImg2ImgPipeline
import sys
import os
import time

def generate_image(pipe, embedding, image_name, diffusion_steps=21):

    embedding = torch.tensor(np.reshape(
        embedding, (1, np.size(embedding))), dtype=torch.float16)
    # print(embedding.size())
    embedding = embedding.to('cuda')

    images = pipe(image_embeds=embedding, num_inference_steps=diffusion_steps).images
    images[0].save(image_name)


def prep_model(model_path):
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16).to('cuda')

    return pipe

if __name__ == "__main__":

    model_path = "../../pretrained_models/stable-diffusion-2-1-unclip"
    pipe = prep_model(model_path)

    root_dir = sys.argv[1]
    embeddings_files_complete = []

    while(True):

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
                generate_image(pipe, embeddings[i,:], filename, diffusion_steps=4)
                finished[i] = 1
                np.savetxt(f"{output_path}/status.txt", finished, delimiter=',')

            embeddings_files_complete.append(difference_list[0])

        elif len(difference_list) > 1:
            raise Exception("Too many embeddings files found!")
        else:
            pass

        time.sleep(1)




