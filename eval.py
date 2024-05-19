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
    parser.add_argument("--prompt", type=str, default= "a nice photo")
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
print('target:', args.target)
print('guidance:', args.guidance)
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
    args.out_dir = '/scratch/gpfs/yg6736'+f'/opt/target{args.target}guidance{args.guidance}seed{args.seed}'
img_dir = args.out_dir + '/images'
try:
    os.makedirs(args.out_dir)
    os.makedirs(img_dir)
except:
    pass

for k  in range(args.opt_steps):
    try:
        os.makedirs(img_dir + f'/optstep_{k}')
    except:
        pass



with open("imagenet_classes.txt", "r") as file:
    imagenet_classes = file.readlines()

imagenet_classes = [class_name.strip() for class_name in imagenet_classes]
sampled_classes = random.sample(imagenet_classes, args.repeat_epoch)



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
    rewards_sum = rewards.sum()
    rewards_sum.backward()
    grads = ims.grad
    biases = - torch.einsum('bijk,bijk->b', grads, ims) + rewards
    #biases = - torch.einsum('bijk,bijk->b', grads, ims)

    return grads, biases, rewards


image_rewards = np.zeros([args.opt_steps, args.repeat_epoch, args.bs])
for n in range(args.repeat_epoch):
    sd_model.set_target(args.target)
    sd_model.set_guidance(args.guidance)
    sd_model.set_linear_reward_model(is_init = True, batch_size = args.bs)
    prompt = sampled_classes[n]
    for k in range(args.opt_steps):
        #sd_model.set_guidance((args.guidance/math.sqrt(k + 1.0)))
        #sd_model.set_guidance((args.guidance/(k + 1.0)))
        if init_latents is None:
            init_i = None
        else:
            init_i = init_latents[n]
        image_, image_eval_ = sd_model(prompt, num_images_per_prompt=args.bs, latents=init_i)

        grads, biases, rewards = get_grad_eval(image_eval_, reward_model)
        grads = grads.clone().detach()
        biases = biases.clone().detach()
        sd_model.set_linear_reward_model(gradients = grads, biases = biases)
        rewards = rewards.detach().cpu().numpy()
        if len(rewards.shape) > 1:
            rewards = rewards.squeeze()
        image_rewards[k, n] = rewards
        
        image_ = image_.images
        for idx, im in enumerate(image_):
            im.save(img_dir + f'/optstep_{k}/{n * args.bs + idx}_reward_{(rewards[idx]).item():.4f}_{prompt}.png')



# calculate mean along batch dimension
image_rewards_mean = np.mean(image_rewards, axis=2)
# calculate std for image_rewards_mean along repeat_epoch dimension
image_rewards_std = np.std(image_rewards_mean, axis=1)
# save std to csv
np.savetxt(args.out_dir + '/rewards_std_in_repeats.csv', image_rewards_std, delimiter=',')

# reshape image_rewards to [opt_steps, repeat_epoch * bs]
image_rewards = image_rewards.reshape(args.opt_steps, -1)
# calculate mean rewards and std
mean_rewards = np.mean(image_rewards, axis=1)
std_rewards = np.std(image_rewards, axis=1)

# save mean_rewards and std_rewards to csv
np.savetxt(args.out_dir + '/mean_rewards.csv', mean_rewards, delimiter=',')
np.savetxt(args.out_dir + '/std_rewards.csv', std_rewards, delimiter=',')
