from gradguided_sdpipeline import GradGuidedSDPipeline
import torch
import numpy as np
from PIL import Image
import PIL
from typing import Callable, List, Optional, Union
from vae import encode
import os
import wandb
import argparse
from scorer import AestheticScorerDiff, RCGDMScorer
import math
import random



def parse():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target", type=float, default=0.)
    parser.add_argument("--guidance", type=float, default=0.) 
    parser.add_argument("--prompt", type=str, default= "")
    parser.add_argument("--out_dir", type=str, default= "")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--opt_steps", type=int, default=100)
    parser.add_argument("--repeat_epoch", type=int, default=1)

    args = parser.parse_args()
    return args



######### preparation ##########

args = parse()

print("-"*50)
print('seed:', args.seed)
print('prompt:', args.prompt)
print('opt_steps:', args.opt_steps)
print('repeat_epoch:', args.repeat_epoch)


device= args.device
save_file = True

## Image Seeds
if args.seed > 0:
    torch.manual_seed(args.seed)
    shape = (args.repeat_epoch, args.bs , 4, 64, 64)
    init_latents = torch.randn(shape, device=device)
else:
    init_latents = None

if args.out_dir == "":
    args.out_dir = '/scratch/gpfs/yg6736'+f'/test_reward/seed{args.seed}_{args.prompt}'

img_dir = args.out_dir + '/images'
high_res_img_dir = args.out_dir + '/high_res_images'
low_res_img_dir = args.out_dir + '/low_res_images'
try:
    os.makedirs(args.out_dir)
    os.makedirs(img_dir)
    os.makedirs(high_res_img_dir)
    os.makedirs(low_res_img_dir)
except:
    pass


prompts = None
if args.prompt != "":
    prompts = [args.prompt] * args.repeat_epoch
else:
    with open("imagenet_classes.txt", "r") as file:
        imagenet_classes = file.readlines()
    imagenet_classes = [class_name.strip() for class_name in imagenet_classes]
    random.seed(args.seed)
    prompts = random.sample(imagenet_classes, args.repeat_epoch)



sd_model = GradGuidedSDPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", local_files_only=True)
sd_model.to(device)


# aesthetic reward model
#reward_model = AestheticScorerDiff().to(device)

# reward of RCGDM
reward_model = RCGDMScorer().to(device)

reward_model.requires_grad_(False)
reward_model.eval()


def get_grad_eval(ims, reward_model):
    ims = ims.to(device)
    ims.requires_grad = True
    rewards = reward_model(ims)
    #print('-'*50)
    #print('rewards shape:', rewards.shape)
    #print('rewards:', rewards)
    #grad_outputs = torch.ones_like(rewards, device=device)
    #rewards.backward(gradient=grad_outputs)
    rewards_sum = rewards.sum()
    rewards_sum.backward()
    grads = ims.grad
    biases = - torch.einsum('bijk,bijk->b', grads, ims) + rewards
    #biases = - torch.einsum('bijk,bijk->b', grads, ims)

    return grads, biases, rewards

sd_model.set_target(args.target)
sd_model.set_guidance(args.guidance)
sd_model.set_linear_reward_model(is_init = True, batch_size = args.bs)


for n in range(args.repeat_epoch):
    prompt = prompts[n]
    if init_latents is None:
        init_i = None
    else:
        init_i = init_latents[n]
    image_, image_eval_ = sd_model(prompt, num_images_per_prompt=args.bs, latents=init_i)
    rewards = reward_model(image_eval_)
    rewards = rewards.detach().cpu().numpy()
    if len(rewards.shape) > 1:
        rewards = rewards.squeeze()
        
        
    image_ = image_.images
    for idx, im in enumerate(image_):
        im.save(img_dir + f'/{n * args.bs + idx}_reward_{(rewards[idx]).item():.4f}_.png')
        if rewards[idx] > 0.2:
            im.save(high_res_img_dir + f'/{n * args.bs + idx}_reward_{(rewards[idx]).item():.4f}_.png')

        if rewards[idx] < -0.5:
            im.save(low_res_img_dir + f'/{n * args.bs + idx}_reward_{(rewards[idx]).item():.4f}_.png')
        