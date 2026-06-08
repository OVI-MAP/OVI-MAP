import os, sys, time, argparse, pickle, logging
from os.path import join as pjoin
from tqdm import tqdm


import cv2, copy
from PIL import Image
import numpy as np
import torch

# image embedding
import clip
from transformers import AutoProcessor, AutoModel


clip_model_list = {
    'clip-ViT-B-32':         'ViT-B/32', 
    'clip-ViT-B-16':         'ViT-B/16', 
    'clip-ViT-L-14':         'ViT-L/14', 
    'clip-ViT-L-14-336px':   'ViT-L/14@336px',
}

siglip_model_list = {
    'siglip-b-16-224':      'google/siglip-base-patch16-224',
    'siglip-b-16-256':      'google/siglip-base-patch16-256',
    'siglip-b-16-384':      'google/siglip-base-patch16-384',
    'siglip-l-16-256':      'google/siglip-large-patch16-256',
    'siglip-l-16-384':      'google/siglip-large-patch16-384',
    'siglip-so400m-14-224': 'google/siglip-so400m-patch14-224',
    'siglip-so400m-14-384': 'google/siglip-so400m-patch14-384',

    'siglip2-b-16-224':      'google/siglip2-base-patch16-224',
    'siglip2-b-16-256':      'google/siglip2-base-patch16-256',
    'siglip2-b-16-384':      'google/siglip2-base-patch16-384',
    'siglip2-l-16-256':      'google/siglip2-large-patch16-256',
    'siglip2-l-16-384':      'google/siglip2-large-patch16-384',
    'siglip2-so400m-14-384': 'google/siglip2-so400m-patch14-384',
}


class VLModel:
    def __init__(self, model_name, img_size, device):
        self.device = device

        # NOTE CLIP and SigLip use PIL.Image in RGB format as input
        if model_name in clip_model_list.keys():
            self.model_type = 'clip'
            self.clip_model, self.prep_clip = clip.load(clip_model_list[model_name], device=device)
        elif model_name in siglip_model_list.keys():
            self.model_type = 'siglip'
            self.siglip_model = AutoModel.from_pretrained(
                siglip_model_list[model_name], device_map="cuda"
            ).eval().to(device)
            self.processor = AutoProcessor.from_pretrained(siglip_model_list[model_name], use_fast=True)
        else:
            raise NotImplementedError

        self.H, self.W = img_size
        self.k_expand = 0.1  # padding factor for the bounding box

    @torch.no_grad()
    def encode_image_with_bbox(self, rgb_img, inst_mask, bbox):
        x1, y1, x2, y2 = bbox

        inst_img = copy.deepcopy(rgb_img)
        inst_img_black_bg = copy.deepcopy(rgb_img)
        inst_img_black_bg[~inst_mask] = 0.0

        batch_img_roi = []
        for layer in range(3):
            x_pad = int(self.k_expand * layer * (x2 - x1))
            y_pad = int(self.k_expand * layer * (y2 - y1))
            x_min = max(0, x1 - x_pad)
            y_min = max(0, y1 - y_pad)
            x_max = min(self.W-1, x2 + x_pad)
            y_max = min(self.H-1, y2 + y_pad)

            if self.model_type == 'clip':
                batch_img_roi.append(
                    self.prep_clip(Image.fromarray(inst_img[y_min:y_max, x_min:x_max]))
                )
                batch_img_roi.append(
                    self.prep_clip(Image.fromarray(inst_img_black_bg[y_min:y_max, x_min:x_max]))
                )
            elif self.model_type == 'siglip':
                batch_img_roi.append(Image.fromarray(inst_img[y_min:y_max, x_min:x_max]))
                batch_img_roi.append(Image.fromarray(inst_img_black_bg[y_min:y_max, x_min:x_max]))

        
        if self.model_type == 'clip': 
            imgs = torch.stack(batch_img_roi).to(self.device)
            roi_feat = self.clip_model.encode_image(imgs)
        elif self.model_type == 'siglip':
            img_inputs = self.processor(images=batch_img_roi, return_tensors="pt").to(device=self.device)
            roi_feat = self.siglip_model.get_image_features(**img_inputs)

        roi_feat = roi_feat / roi_feat.norm(dim=-1, keepdim=True)
        roi_feat = roi_feat.mean(dim=0).detach().cpu().numpy()

        return roi_feat