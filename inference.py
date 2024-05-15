
from sd_pipeline import GuidedSDPipeline
import torch
import numpy as np
from PIL import Image
import PIL
from typing import Callable, List, Optional, Union
from dataset import CustomCIFAR10Dataset, CustomLatentDataset
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

    args = parser.parse_args()
    return args


######### preparation ##########

args = parse()
device= args.device
save_file = True
#reward_model_file='convnet.pth'

## Image Seeds
if args.seed > 0:
    torch.manual_seed(args.seed)
    shape = (args.num_images//args.bs, args.bs , 4, 64, 64)
    init_latents = torch.randn(shape, device=device)
else:
    init_latents = None

if args.out_dir == "":
    args.out_dir = f'imgs/target{args.target}guidance{args.guidance}_{args.prompt}'
try:
    os.makedirs(args.out_dir)
except:
    pass

wandb.init(project="guided_dm", config={
    'target': args.target,
    'guidance': args.guidance, 
    'prompt': args.prompt,
    'num_images': args.num_images
})


sd_model = GuidedSDPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", local_files_only=True)
sd_model.to(device)


# aesthetic reward model
reward_model = AestheticScorerDiff().to(device)
reward_model.requires_grad_(False)
reward_model.eval()

sd_model.setup_reward_model(reward_model)
sd_model.set_target(args.target)
sd_model.set_guidance(args.guidance)

image = []
image_eval = []
for i in range(args.num_images // args.bs):
    if init_latents is None:
        init_i = None
    else:
        init_i = init_latents[i]
    image_, image_eval_ = sd_model(args.prompt, num_images_per_prompt=args.bs, latents=init_i) # List of PIL.Image objects
    image_ = image_.images
    image.extend(image_)
    image_eval.append(image_eval_)


if save_file:
    for idx, im in enumerate(image):
        im.save(args.out_dir +'/'+ f'{idx}_gt_pred_.png')


###### evaluation and metric #####

#gt_dataset = CustomCIFAR10Dataset(image)
#gt_dataloader = torch.utils.data.DataLoader(gt_dataset, batch_size=20, shuffle=False, num_workers=8)

#pred_dataset = CustomLatentDataset(image)
#pred_dataloader = torch.utils.data.DataLoader(pred_dataset, batch_size=20, shuffle=False, num_workers=8)

ground_truth_reward_model = AestheticScorerDiff().to(device)
ground_truth_reward_model.requires_grad_(False)
ground_truth_reward_model.eval()

rewards = []
with torch.no_grad():
    total_reward_gt = []
    for idx, input in  enumerate(image_eval):
        input = input.to(device)
        gt_reward = ground_truth_reward_model(input)
        #print(gt_rewards, torch.mean(gt_rewards))
        rewards.append( gt_reward.cpu().numpy())
        image = (input.clone().detach() / 2 + 0.5).clamp(0, 1)
        pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        pil = pil.resize((256, 256))
        pil.save(args.out_dir +'/'+ f'{idx}_latent_reward_{rewards[idx]:.4f}_.png')
print("_"*50)
print(rewards)

    #total_reward_gt = np.concatenate(total_reward_gt, axis=None)

    #wandb.log({"gt_reward_mean": np.mean(total_reward_gt) ,
     #          "gt_reward_std": np.std(total_reward_gt) })


#with torch.no_grad():
#    total_reward_pred= []
#    for inputs in pred_dataloader:
#        inputs = inputs.to(device)
#        inputs = encode(inputs)
 #       pred_rewards = reward_model(inputs)
        #print(pred_rewards, torch.mean(pred_rewards))
 #       total_reward_pred.append(pred_rewards.cpu().numpy())

 #   total_reward_pred = np.concatenate(total_reward_pred, axis=None)
#    wandb.log({"pred_reward_mean": np.mean(total_reward_pred) ,
 #              "pred_reward_std": np.std(total_reward_pred) })


#if save_file:
#    for idx, im in enumerate(image):
#        im.save(args.out_dir +'/'+ f'{idx}_gt_{total_reward_gt[idx]:.4f}_pred_{total_reward_pred[idx]:.4f}.png')
