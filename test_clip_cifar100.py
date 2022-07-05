import clip
import torch
import PIL
import numpy as np
from tqdm import tqdm
import os
from torchvision.datasets import CIFAR100
import typer
from time import time

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

device = "mps" if torch.has_mps else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def encode_text(model, text):
    x = model.token_embedding(text).type(model.dtype)  # [batch_size, n_ctx, d_model]

    x = x + model.positional_embedding.type(model.dtype)
    num_bpe_tokens, num_text, embedding_size = x.shape
    x = x.transpose(1,0).contiguous().view(num_text, num_bpe_tokens, embedding_size)  # NLD -> LND
    x = model.transformer(x)
    x = x.transpose(1,0).contiguous().view(num_bpe_tokens, num_text, embedding_size) # LND -> NLD
    x = model.ln_final(x).type(model.dtype)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    try:
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ model.text_projection
    except NotImplementedError:
        text = text.to('cpu')
        x = x.to('cpu')
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)].to('mps') @ model.text_projection
    return x


def encode_image(vision_model, x: torch.Tensor):
    x = vision_model.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    batch_size, width, grid = x.shape
    x = x.transpose(2,1).contiguous().view(batch_size, grid, width)#x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat([vision_model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + vision_model.positional_embedding.to(x.dtype)
    x = vision_model.ln_pre(x)

    batch_size, width, grid = x.shape
    x = x.transpose(1,0).contiguous().view(width, batch_size, grid) #x.permute(1, 0, 2)  # NLD -> LND
    x = vision_model.transformer(x)
    x = x.transpose(1,0).contiguous().view(batch_size, width, grid)#x.permute(1, 0, 2)  # LND -> NLD

    x = vision_model.ln_post(x[:, 0, :])

    if vision_model.proj is not None:
        x = x @ vision_model.proj

    return x

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
    print(f"Preprocessing time {time()-start_time}")
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

    return image_input, text_inputs

def get_text_features(text_inputs):
    # Calculate text features
    with torch.no_grad():
        text_features = encode_text(model, text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

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
