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
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from meshgpt_pytorch import (
    MeshTransformerTrainer,
    MeshAutoencoderTrainer,
    MeshAutoencoder,
    MeshTransformer
)
from meshgpt_pytorch.data import derive_face_edges_from_faces


def load_obj_from_json(path):
    obj_datas = []
    with open(path, "r") as json_file:
        loaded_data = json.load(json_file) 
        for item in loaded_data:
            # if len(item['faces']) >= 512:
            #     continue
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
    tables = load_obj_from_json("/f_ndata/zekai/ShapeNetCore.v2/table_noaug.json")
    dataset = MeshDataset(tables) 
    return dataset

def load_autoencoder(): 
    autoencoder = MeshAutoencoder( 
        dim = 512,
        use_residual_lfq = True,
        commit_loss_weight = 0.1,
        bin_smooth_blur_sigma = 0.4
    )   
    return autoencoder

def load_gpt2(autoencoder):
    max_length =  max(len(d["faces"]) for d in dataset if "faces" in d) 
    max_seq =  max_length * 6  
    # print(f'GPT2 max length: {max_seq}')
    # max_seq = 512*6
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


def check_tokenize(autoencoder):
    data_sample = dataset[0]

    codes = autoencoder.tokenize(
        vertices = data_sample['vertices'],
        faces = data_sample['faces']
    )

    codes = codes.reshape(data_sample['faces'].shape[0],3,2)
    print(f"Codes: {codes.shape}")

    continuous_coors, pred_face_coords, face_mask = autoencoder.decode_from_codes_to_faces(mesh_token_ids, return_discrete_codes = True)
    pred_face_coords = pred_face_coords.squeeze(1)
    print(f"Label Faces: {face_coordinates.shape}")
    print(face_coordinates)
    print(f"Pred Faces: {pred_face_coords.shape}")
    print(pred_face_coords)

check_tokenize(autoencoder)

# # train autoencoder
# autoencoder_trainer = MeshAutoencoderTrainer(
#     model = autoencoder,
#     learning_rate = 1e-3, 
#     warmup_steps = 10,
#     dataset = dataset,  
#     checkpoint_every_epoch = 20, 
#     batch_size=16,
#     grad_accum_every=1,
#     num_train_steps = 100
# )

# loss = autoencoder_trainer.train(num_epochs = 160)   
# autoencoder_trainer.save(f'checkpoints/autoencoder_final.pt') 


# # train gpt2
# gpt_trainer = MeshTransformerTrainer(
#     model = transformer,
#     learning_rate = 5e-4, 
#     warmup_steps = 10,
#     dataset = dataset,  
#     checkpoint_every_epoch = 20, 
#     batch_size=4,
#     grad_accum_every=16,
#     num_train_steps = 100
# ) 
# loss = gpt_trainer.train(num_epochs = 160)  

# gpt_trainer.save(f'checkpoints/autoencoder_final.pt')



def generate(transformer, path):
    obj = ""
    coords = transformer.generate() 
    tensor_data = coords[0].cpu()
    numpy_data = tensor_data.numpy().reshape(-1, 3)
    for v in numpy_data:
        obj += f"v {v[0]} {v[1]} {v[2]}\n"
    for i in range(1, len(numpy_data), 3):
        obj += f"f {i} {i + 1} {i + 2}\n"

    with open(path, "w") as f:
        f.write(obj)
    
# generate(transformer, 'test_generate.obj')