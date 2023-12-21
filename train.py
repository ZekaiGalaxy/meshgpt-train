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

from meshgpt_pytorch import (
    MeshTransformerTrainer,
    MeshAutoencoderTrainer,
    MeshAutoencoder,
    MeshTransformer
)

def load_json(file,num_examples):
    obj_datas = []
    with open(file, "r") as json_file:
        loaded_data = json.load(json_file) 
        for item in loaded_data:
            for _ in range(num_examples):
                obj_data = {"vertices": torch.tensor(item["vertices"], dtype=torch.float), "faces":  torch.tensor(item["faces"], dtype=torch.long),"texts": item["texts"] } 
                obj_datas.append(obj_data)
    return obj_datas

import json
class MeshDataset(Dataset): 
    def __init__(self, data): 
        self.data = data
        print(f"Got {len(data)} data")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        return self.data[idx]
    
    def embed_texts(self,transformer): 
        unique_texts = set(item['texts'] for item in self.data)
 
        text_embeddings = transformer.embed_texts(list(unique_texts))
        print(f"Got text_embeddings: {len(text_embeddings)}") 
        text_embedding_dict = dict(zip(unique_texts, text_embeddings))
 
        for item in self.data:
            text_value = item['texts']
            item['text_embeds'] = text_embedding_dict.get(text_value, None)
            del item['texts']
 
        
    def sample_obj(self):
        all_vertices = []
        all_faces = []
        vertex_offset = 0 


        translation_distance = 0.5  # Adjust as needed 
        vertex_offset = len(all_vertices)
        
        for r, faces_coordinates in enumerate(self.data):    
            if r > 30:
                break
            for vertex in faces_coordinates["vertices"]: 
                all_vertices.append(f"v {vertex[0]+translation_distance * (r / 0.2 - 1)} {vertex[1]} {vertex[2]}\n")
                #all_vertices.append(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

            for face in faces_coordinates["faces"]:
                all_faces.append(f"f {face[0]+1+vertex_offset} {face[1]+1+vertex_offset} {face[2]+1+vertex_offset}\n") 
                try:
                    all_vertices[face[0]+vertex_offset]
                    all_vertices[face[1]+vertex_offset]
                    all_vertices[face[2]+vertex_offset]
                except Exception  as e :
                    print(e)
                    print(face[0]+vertex_offset)
                    print(face[1]+vertex_offset)
                    print(face[2]+vertex_offset)
                    print(len(all_vertices))
                
            vertex_offset = len(all_vertices) 


        obj_file_content = "".join(all_vertices) + "".join(all_faces)

        # Save to a single file
        obj_file_path = "./combined_3d_models.obj"
        with open(obj_file_path, "w") as file:
            file.write(obj_file_content)

        print(obj_file_path)

tables = load_json("/f_ndata/zekai/ShapeNetCore.v2/table.json",1)
dataset = MeshDataset(tables) 

autoencoder = MeshAutoencoder( 
    dim = 512
) 
total_params = sum(p.numel() for p in autoencoder.encoders.parameters())
print(f"encoders Total parameters: {total_params}")
total_params = sum(p.numel() for p in autoencoder.decoders.parameters())
print(f"decoders Total parameters: {total_params}")  

total_params = sum(p.numel() for p in autoencoder.encoders.parameters())
print(f"Total parameters: {total_params}")
print(autoencoder.encoders)

autoencoder_trainer = MeshAutoencoderTrainer(model =autoencoder,learning_rate = 1e-3, 
                                             warmup_steps = 10,
                                             dataset = dataset,   
                                             num_train_steps=100,
                                             batch_size=16,
                                             grad_accum_every=1)

loss = autoencoder_trainer.train(40,stop_at_loss = 0.25)   
autoencoder_trainer = MeshAutoencoderTrainer(model =autoencoder,learning_rate = 1e-4, 
                                             warmup_steps = 10,
                                             dataset = dataset,
                                             checkpoint_every_epoch = 20,  
                                             num_train_steps=100,
                                             batch_size=16,
                                             grad_accum_every=1)

loss = autoencoder_trainer.train(200,stop_at_loss = 0.25)   
autoencoder_trainer.save(f'mesh-encoder_4_loss_{loss:.3f}.pt') 