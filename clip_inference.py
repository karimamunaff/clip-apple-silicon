import clip
import torch
from time import time

if torch.has_mps:
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

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
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)].to(device) @ model.text_projection
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