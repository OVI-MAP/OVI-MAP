import os, time, argparse, logging, pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

import clip
from transformers import AutoModel, AutoTokenizer

FORMAT = '%(asctime)s.%(msecs)06d %(levelname)-8s: [%(filename)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%H:%M:%S')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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


def init_label_colormap(task='Replica'):
    """
    Return:
        text_prompt: The text prompt for the task
        labels: The list of text labels for the task
        color_map: The color map for the task, from text label to RGB
        valid_ids: The list of valid IDs for the task
    """
    if task == 'NYU40':
        text_prompt = 'NYU41'
        from semantic_const import NYU_40, NYU_41_PALETTE
        labels = ['background'] + NYU_40
        valid_ids = list(range(len(labels)))
        color_map = {label: NYU_41_PALETTE[i] for i, label in enumerate(labels)}
    elif task == 'CoCoPano':
        text_prompt = 'COCO134'
        from semantic_const import COCO_PANOPTIC_133, COCO_PANOPTIC_134_PALETTE
        labels = ['background'] + COCO_PANOPTIC_133
        valid_ids = list(range(len(labels)))
        color_map = {label: COCO_PANOPTIC_134_PALETTE[i] for i, label in enumerate(labels)}
    elif task == 'Replica':
        text_prompt = 'Replica'
        from semantic_const import REPLICA_51, REPLICA_52_PALETTE
        labels = ['background'] + REPLICA_51
        valid_ids = list(range(len(labels)))
        color_map = {label: REPLICA_52_PALETTE[i] for i, label in enumerate(labels)}
    elif task == 'Scannet200':
        text_prompt = 'Scannet200'
        from semantic_const import CLASS_LABELS_200, SCANNET_COLOR_MAP_200, VALID_CLASS_IDS_200
        labels = ['background'] + CLASS_LABELS_200
        valid_ids = [0] + VALID_CLASS_IDS_200
        color_map = {labels[i]: SCANNET_COLOR_MAP_200[id] for i, id in enumerate(valid_ids)}
    elif task == 'Scannet20':
        text_prompt = 'Scannet20'
        from semantic_const import CLASS_LABELS_20, SCANNET_COLOR_MAP_20, VALID_CLASS_IDS_20
        labels = ['background'] + CLASS_LABELS_20
        valid_ids = [0] + VALID_CLASS_IDS_20
        color_map = {labels[i]: SCANNET_COLOR_MAP_20[id] for i, id in enumerate(valid_ids)}
    else:
        raise NotImplementedError(f'Unknown task: {task}')
    
    return text_prompt, labels, color_map, valid_ids



class TextEmbedder:
    def __init__(self, model_name, device, lazy_init=False):
        self.model_name = model_name
        self.device = device
        self.canonical_phrases = ['object', 'things', 'stuff', 'texture']
        if lazy_init:
            return

        self.load_model(model_name, device)

    def load_model(self, model_name, device):
        if model_name in clip_model_list.keys():
            self.model_type = 'clip'
            self.clip_model, _ = clip.load(clip_model_list[model_name], device=device)
        elif model_name in siglip_model_list.keys():
            self.model_type = 'siglip'
            self.siglip_model = AutoModel.from_pretrained(siglip_model_list[model_name]).eval().to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(siglip_model_list[model_name])
        else:
            raise NotImplementedError(f'Unknown model name: {model_name}')
   

    @torch.no_grad()
    def get_text_embed(self, text_cands, load_np=None):
        if load_np is not None:
            text_embed = np.load(load_np)
            return text_embed.to(self.device)

        if self.model_type == 'clip':
            text_inputs = clip.tokenize(text_cands).to(self.device)
            text_feats = self.clip_model.encode_text(text_inputs)
        elif self.model_type == 'siglip':
            text_inputs = self.tokenizer(text_cands, padding="max_length", 
                max_length=64, return_tensors="pt").to(self.device)
            text_feats = self.siglip_model.get_text_features(**text_inputs)
        return text_feats

    @torch.no_grad()
    def get_canonical_text_embed(self, load_np=None):
        if load_np is not None:
            canonical_text_embed = np.load(load_np)
            return canonical_text_embed.to(self.device)

        return self.get_text_embed(self.canonical_phrases)


# python -m scripts.utils.text_embedding

def main():

    # result_dir = '/home/zilong/Disk_data/semantic_mapping_result'
    # os.makedirs(result_dir, exist_ok=True)

    # choose from the model list above
    # clip-ViT-L-14-336px
    # siglip-so400m-14-384 | siglip-l-16-384
    # siglip2-l-16-384 | siglip2-so400m-14-384
    model_name = 'siglip2-so400m-14-384'
    save_npy = False

    # NYU40 | CoCoPano | CoCo134 | Replica | Scannet200 | Scannet20
    task = 'Replica' 
    print(f'Using model: {model_name}, text prompts: {task}')
    text_prompt, text_cands, _, _ = init_label_colormap(task)
    test_embeder = TextEmbedder(model_name, device=DEVICE)

    if save_npy:
        text_feats_canon = test_embeder.get_canonical_text_embed().detach().cpu().to(torch.float32).numpy()
        np.save(os.path.join('.', f'{model_name}_canonical.npy'), text_feats_canon)
    else:
        # prompt_text = [f'This is a photo of a {t}' for t in text_cands]
        text_feats = test_embeder.get_text_embed(text_cands)
        text_feats = text_feats.detach().cpu().to(torch.float32).numpy()
        text_embed = {}
        for i in range(text_feats.shape[0]):
            text_embed[text_cands[i]] = text_feats[i]
        save_pkl_path = os.path.join('.', f'{model_name}_{text_prompt}.pkl')
        with open(save_pkl_path, 'wb') as f:
            pickle.dump(text_embed, f)

    # print(f'Max VMem {torch.cuda.max_memory_allocated() / 1024**3:.6f} GB')

if __name__ == '__main__':
    main()