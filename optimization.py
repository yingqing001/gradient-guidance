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
#print("aesthetic reward optimization")
print('seed:', args.seed)
print('target:', args.target)
print('guidance:', args.guidance)
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
    args.out_dir = '/scratch/gpfs/yg6736'+f'/opt_test/target_{11366}_{args.target}guidance{args.guidance}seed{args.seed}_{args.prompt}'

img_dir = args.out_dir + '/images'
try:
    os.makedirs(args.out_dir)
    os.makedirs(img_dir)
except:
    pass

wandb.init(project="gradient_guided_dm", name=f'test_target{args.target}guidance{args.guidance}seed{args.seed}_{args.prompt}',
    config={
    'target': args.target,
    'guidance': args.guidance, 
    'seed': args.seed
})

prompts = None
if args.prompt != "":
    prompts = [args.prompt] * args.repeat_epoch
else:
    with open("imagenet_classes.txt", "r") as file:
        imagenet_classes = file.readlines()
    imagenet_classes = [class_name.strip() for class_name in imagenet_classes]
    random.seed(args.seed)
    prompts = random.sample(imagenet_classes, args.repeat_epoch)

targets = [0, 1, 1, 3] + [args.target] * args.opt_steps
guidances = [args.guidance] * args.opt_steps


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


image_rewards = np.zeros([args.opt_steps, args.repeat_epoch, args.bs])
for n in range(args.repeat_epoch):
    sd_model.set_target(args.target)
    sd_model.set_guidance(args.guidance)
    sd_model.set_linear_reward_model(is_init = True, batch_size = args.bs)
    prompt = prompts[n]
    for k in range(args.opt_steps):
        #sd_model.set_guidance((args.guidance/math.sqrt(k + 1.0)))
        #sd_model.set_guidance((args.guidance/(k + 1.0)))
        sd_model.set_target(targets[k])
        sd_model.set_guidance(guidances[k])
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
        wandb.log({'step': k, 'reward_mean': rewards.mean(), 'reward_std': rewards.std()})

        image_ = image_.images
        for idx, im in enumerate(image_):
            im.save(img_dir + f'/{n * args.bs + idx}_optstep_{k}_reward_{(rewards[idx]).item():.4f}_.png')

# calculate mean along batch dimension
image_rewards_mean = np.mean(image_rewards, axis=2)
# save mean to csv
np.savetxt(args.out_dir + '/rewards_mean.csv', image_rewards_mean, delimiter=',')
# calculate std for image_rewards_mean along repeat_epoch dimension
image_rewards_std = np.std(image_rewards_mean, axis=1)
# save std to csv
np.savetxt(args.out_dir + '/rewards_std_in_repeats.csv', image_rewards_std, delimiter=',')

# reshape image_rewards to [opt_steps, repeat_epoch * bs]
image_rewards = image_rewards.reshape(args.opt_steps, -1)
# save image_rewards to csv opt_steps x (repeat_epoch * bs)
np.savetxt(args.out_dir + '/rewards.csv', image_rewards, delimiter=',')


# calculate mean rewards and std
mean_rewards = np.mean(image_rewards, axis=1)
std_rewards = np.std(image_rewards, axis=1)

# save mean_rewards and std_rewards to csv
np.savetxt(args.out_dir + '/mean_rewards.csv', mean_rewards, delimiter=',')
np.savetxt(args.out_dir + '/std_rewards.csv', std_rewards, delimiter=',')

#for k in range(args.opt_steps):
 #   wandb.log({'mean_reward_total': mean_rewards[k], 'std_reward_total': std_rewards[k]})

## plot mean rewards and std error bar
import matplotlib.pyplot as plt
x = np.arange(args.opt_steps)
plt.figure(figsize=(10, 6))
plt.plot(x, mean_rewards, color='dodgerblue')
plt.fill_between(x, mean_rewards - std_rewards, mean_rewards + std_rewards, color='dodgerblue', alpha=0.2)
plt.xlabel('Optimization Steps')
plt.ylabel('Reward')
# save
plt.savefig(args.out_dir + '/reward_plot.png')
plt.close()


# plot mean_rewards and use image_rewards_std as error bar
plt.figure(figsize=(10, 6))
# plot mean_rewards curve
plt.plot(x, mean_rewards, color='dodgerblue')
plt.errorbar(x, mean_rewards, yerr=image_rewards_std, fmt='o', color='dodgerblue')
plt.xlabel('Optimization Steps')
plt.ylabel('Reward')
# save
plt.savefig(args.out_dir + '/reward_plot_errorbar.png')
plt.close()
