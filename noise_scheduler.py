import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from HeightmapDataset import HeightmapDataset

BATCH_SIZE = 64 #128
IMG_SIZE = 64

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    #print("shape of t from get index: ", t.shape)
    batch_size = t.shape[0] # references the first element in shape()
    #print("batch_size from get index: ", batch_size)
    out = vals.gather(-1, t.cpu())
    #print("out: ", out)
    #print("x_shape from get index:", x_shape)
    #print("(1,) * (len(x_shape) - 1): ", (1,) * (len(x_shape) - 1))
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device = "cuda"):
    """
    Takes an image and timestep as input to return a noisy version of it
    Device = Change device to cpu or cuda(if you have a cuda graphics card)
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    
    #print("x_0 size IMAGE: ", x_0.size())
    #print("x_0 IMAGE: ", x_0)

    #print("Sqrt alphas RESULT OF GETINDEX: ", sqrt_alphas_cumprod_t.size())
    #print("Sqrt alphas RESULT OF GETINDEX: ", sqrt_alphas_cumprod_t)
    #Maybe try iterating through the sqrt thingo
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

def show_tensor_image(image):
    #print(image.size())
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.detach().cpu().numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    #print(reverse_transforms(image))
    plt.imshow(reverse_transforms(image))

def get_loss(model, x_0, t, device = "cuda"):
    #print("x_0 from get loss: ", x_0) 
    #print("t from get loss: ", t)
    x_noisy, noise = forward_diffusion_sample(x_0, t, device) #forward_diffusion_sample(x_0, t, device = "cuda")
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T)
#print(betas)
# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

data = HeightmapDataset(dir= 'resources/', transform = [
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(), # Scales data into [0, 1]
    transforms.Lambda(lambda t: (t * 2) - 1) # Scales between [-1, 1]
    ])
dataloader = DataLoader(data, batch_size = BATCH_SIZE, shuffle = True)

# image = next(iter(dataloader))[0]

# plt.figure(figsize=(15,15))
# plt.axis('off')
# num_images = 10
# stepsize = int(T/num_images)

# for idx in range(0, T, stepsize):
#     t = torch.Tensor([idx]).type(torch.int64)
#     #print("the t: ", t)
#     plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
#     image, noise = forward_diffusion_sample(image, t)
#     show_tensor_image(image)
# plt.show()