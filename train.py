# No Augmentation, Overfit Table
import torch
import trimesh
import numpy as np
import os
import csv 
import json
import math 
import torch
from torch.utils.data import Dataset, DataLoader 
from tqdm import tqdm
import numpy as np
import torch
import json

from meshgpt_pytorch import (
    MeshTransformerTrainer,
    MeshAutoencoderTrainer,
    MeshAutoencoder,
    MeshTransformer
)


def load_obj_from_json(path):
    obj_datas = []
    with open(path, "r") as json_file:
        loaded_data = json.load(json_file) 
        for item in loaded_data:
            obj_data = {"vertices": torch.tensor(item["vertices"], dtype=torch.float), "faces":  torch.tensor(item["faces"], dtype=torch.long),"texts": item["texts"]} 
            obj_datas.append(obj_data)
    return obj_datas


class MeshDataset(Dataset): 
    def __init__(self, data): 
        self.data = data
        print(f"Got {len(data)} data")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        return self.data[idx]

def load_data():
    tables = load_obj_from_json("/f_ndata/zekai/ShapeNetCore.v2/table.json")
    dataset = MeshDataset(tables) 
    return dataset

def load_autoencoder(): 
    autoencoder = MeshAutoencoder( 
        dim = 512,
        use_residual_lfq = False,
        commit_loss_weight = 0.01
    )   
    return autoencoder

def load_gpt2(autoencoder):
    max_length =  max(len(d["faces"]) for d in dataset if "faces" in d) 
    max_seq =  max_length * 6  
    print(f'GPT2 max length: {max_seq}')
    transformer = MeshTransformer(
        autoencoder,
        dim = 768,
        max_seq_len = max_seq,
        condition_on_text = False
    )
    return transformer

dataset = load_data()
autoencoder = load_autoencoder()
transformer = load_gpt2(autoencoder)

# train autoencoder
autoencoder_trainer = MeshAutoencoderTrainer(
    model = autoencoder,
    learning_rate = 1e-4, 
    warmup_steps = 10,
    dataset = dataset,  
    checkpoint_every_epoch = 20, 
    batch_size=16,
    grad_accum_every=4,
    use_wandb_tracking=True
)

loss = autoencoder_trainer.train(num_epochs = 1)   
autoencoder_trainer.save(f'checkpoints/autoencoder_final.pt') 


# train gpt2
gpt_trainer = MeshTransformerTrainer(
    model = transformer,
    learning_rate = 1e-4, 
    warmup_steps = 10,
    dataset = dataset,  
    checkpoint_every_epoch = 20, 
    batch_size=16,
    grad_accum_every=4,
    use_wandb_tracking=True
) 
loss = gpt_trainer.train(num_epochs = 1)  

gpt_trainer.save(f'checkpoints/autoencoder_final.pt')



def generate(transformer, path):
    obj = ""
    coords = transformer.generate() 
    tensor_data = coords[0].cpu()
    numpy_data = tensor_data.numpy().reshape(-1, 3)
    
    for v in numpy_data:
        obj += f"v {v[0]} {v[1]} {v[2]}\n"

    for i in range(1, len(numpy_data), 3):
        obj += f"f {i} {i + 1} {i + 2}\n"

    # path = f'./tests/3d_output_{text}.obj'
    with open(path, "w") as f:
        f.write(obj)