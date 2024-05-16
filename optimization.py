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
from aesthetic_scorer import AestheticScorerDiff



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
print('prompt:', args.prompt)


device= args.device
save_file = True

## Image Seeds
if args.seed > 0:
    torch.manual_seed(args.seed)
    shape = (args.repeat_ephoch, args.bs , 4, 64, 64)
    init_latents = torch.randn(shape, device=device)
else:
    init_latents = None

if args.out_dir == "":
    args.out_dir = f'opt/target{args.target}guidance{args.guidance}seed{args.seed}_{args.prompt}'
img_dir = args.out_dir + '/images'
try:
    os.makedirs(args.out_dir)
    os.makedirs(img_dir)
except:
    pass

wandb.init(project="guided_dm", config={
    'target': args.target,
    'guidance': args.guidance, 
    'prompt': args.prompt,
    'num_images': args.num_images
})


sd_model = GradGuidedSDPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", local_files_only=True)
sd_model.to(device)


# aesthetic reward model
reward_model = AestheticScorerDiff().to(device)
reward_model.requires_grad_(False)
reward_model.eval()

def get_grad_eval(ims, reward_model):
    ims = ims.to(device)
    ims.requires_grad = True
    rewards = reward_model(ims)
    rewards.backward()
    grads = ims.grad
    biases = - grads * ims + rewards

    return grads, biases, rewards


image_rewards = np.zeros([args.opt_steps, args.repeat_epoch, args.bs])
for n in range(args.repeat_epoch):
    sd_model.set_target(args.target)
    sd_model.set_guidance(args.guidance)
    sd_model.set_linear_reward_model(is_init = True, batch_size = args.bs)
    for k in range(args.opt_steps):
        if init_latents is None:
            init_i = None
        else:
            init_i = init_latents[n]
        image_, image_eval_ = sd_model(args.prompt, num_images_per_prompt=args.bs, latents=init_i)

        grads, biases, rewards = get_grad_eval(image_eval_, reward_model)
        grads = grads.clone().detach()
        biases = biases.clone().detach()
        sd_model.set_linear_reward_model(grads = grads, biases = biases)
        rewards = rewards.cpu().numpy()
        image_rewards[k, n] = rewards

        image_ = image_.images
        for idx, im in enumerate(image_):
            im.save(img_dir + f'/{n * args.bs + idx}_optstep_{k}_reward_{(rewards[idx]).item():.4f}_.png')


mean_rewards = image_rewards.mean(axis=(1, 2))

## plot mean rewards
import matplotlib.pyplot as plt
plt.plot(mean_rewards)
plt.xlabel('Optimization Steps')
plt.ylabel('Reward')
plt.savefig(args.out_dir + '/reward_plot.png')

wandb.log({'mean_reward': mean_rewards[-1]})
wandb.finish()
