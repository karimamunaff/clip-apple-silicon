import clip
import torch
from tqdm import tqdm
import os
from torchvision.datasets import CIFAR100
import typer
from time import time
from clip_inference import *

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

def get_inputs():
    start_time = time()
    # Prepare the inputs
    images, class_ids = [],[]
    for image_index in range(10000):
        try:
            image, class_id  = cifar100[image_index]
            images.append(image)
            class_ids.append(class_id)
        except TypeError:
            continue
    image_input = torch.cat([preprocess(image).unsqueeze(0) for image in images]).to(device)
    print(image_input.shape)
    print(f"Preprocessing time {time()-start_time}")
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

    return image_input, text_inputs
    
def run(batch_size:int = 64):

    image_input, text_inputs = get_inputs()
    text_features = get_text_features(text_inputs)
    
    pbar = tqdm(total = len(image_input))

    for image_batch in batch(image_input, batch_size):
        # Calculate features
        with torch.no_grad():
            image_features = encode_image(model.visual, image_batch.type(model.dtype))

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        pbar.update(len(image_batch))
    pbar.close()


if __name__ == "__main__":
    typer.run(run)
