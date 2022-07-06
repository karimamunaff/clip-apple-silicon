from clip_inference import *
from pathlib import Path
from tqdm import tqdm
import PIL
import typer
from time import time
import ray
ray.init(num_gpus=1, num_cpus=9)  

@ray.remote
def preprocess_image(image_path):
    image = PIL.Image.open(image_path)
    return preprocess(image).unsqueeze(0)

def get_images(image_paths):
    preprocessed_images = []
    for path in image_paths:
        preprocessed_images.append(preprocess_image.remote(path))
    preprocessed_images = ray.get(preprocessed_images)
    return torch.cat(preprocessed_images).to(device)

def get_image_features(preprocessed_images):
    with torch.no_grad():
        image_features = encode_image(model.visual, preprocessed_images.type(model.dtype))
    return image_features

def run(image_dir:str, batch_size:int, num_images:int=10000):
    preprocessing_time = 0
    image_paths = list(Path(image_dir).rglob("*.jpg"))[:num_images]
    pbar = tqdm(total = len(image_paths))
    pbar.set_description("Computing Image Features: ")
    for paths_batch in batch(image_paths, batch_size):
        start_time = time()
        preprocessed_images = get_images(paths_batch)
        preprocessing_time += (time()-start_time)
        get_image_features(preprocessed_images)
        pbar.update(len(paths_batch))
    pbar.close()
    print(f"Preprocessing time: {preprocessing_time}")
    
if __name__ == "__main__":
    typer.run(run)
    