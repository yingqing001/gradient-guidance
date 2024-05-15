from importlib import resources
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPModel, CLIPProcessor
import torchvision
from PIL import Image
ASSETS_PATH = resources.files("asset")

class MLPDiff(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )


    def forward(self, embed):
        return self.layers(embed)


class AestheticScorerDiff(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLPDiff()
        state_dict = torch.load(ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth"))
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()
                                               
    def __call__(self, images):
        #target_size = 224
        #normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
        #                                                   std=[0.26862954, 0.26130258, 0.27577711])
        
        if isinstance(images, torch.Tensor):
            print(images)
            images = (images + 1) / 2.0
            images = images.clamp(0, 1)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
            print(images)
            #print('-------------------')
            #print("images shape:")
            #print(images.shape)
        else:
            #print('++++++++++++++++++++')
            #print("images shape:")
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)

        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)

        
    


def aesthetic_reward_fn(device=None):
    target_size = 224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    scorer = AestheticScorerDiff().to(device)
    scorer.requires_grad_(False)
    scorer.eval()
    def reward_fn(im_pix_un):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        im_pix = torchvision.transforms.Resize(target_size)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)
        rewards = scorer(im_pix)
        return rewards
    
    return reward_fn
    