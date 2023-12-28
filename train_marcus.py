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
from collections import OrderedDict

# chair 1*500
# (16,0.1,20); (16,0.1,5)
# chair 10*50
# (16,0.1,40)+(16,0.001,20); (16,0.1,10)+(16,0.005,15)
# chair 50*100 # 100 is important
# (64,0.01) -> 0.01 is important, can bring to 0.7



class MeshDataset(Dataset): 
    def __init__(self, data): 
        self.data = data
        print(f"Got {len(data)} data")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        return self.data[idx]

    def save(self, path):
        np.savez_compressed(path, *self.data)

    @classmethod
    def load(cls, path): 
        loaded_data = np.load(path, allow_pickle=True)
 
        data = []
        for i in range(len(loaded_data.files)):
            data_item = {}
            for key in loaded_data[f"arr_{i}"].item():
                data_item[key] = loaded_data[f"arr_{i}"].item()[key]
            data.append(data_item)

        return cls(data)

    def embed_texts(self,transformer): 
        unique_texts = set(item['texts'] for item in self.data)
 
        text_embeddings = transformer.embed_texts(list(unique_texts))
        print(f"Got text_embeddings: {len(text_embeddings)}") 
        text_embedding_dict = dict(zip(unique_texts, text_embeddings))
 
        for item in self.data:
            text_value = item['texts']
            item['text_embeds'] = text_embedding_dict.get(text_value, None)
            del item['texts']

    def generate_face_edges(self):
        n = 0
        for i in range(0, len(self.data)):  
            item = self.data[i]
            item['face_edges'] =  derive_face_edges_from_faces(item['faces'])
            n += 1  
        print(f"done {n}/{len(self.data)}")

    def generate_codes(self, autoencoder : MeshAutoencoder):
        n = 0
        for i in range(0, len(self.data)):  
            item = self.data[i]
             
            codes = autoencoder.tokenize(
                vertices = item['vertices'],
                faces = item['faces'],
                face_edges = item['face_edges']
            ) 
            item['codes'] = codes 
            n += 1  

        print(f"[generate_codes] done {n}/{len(self.data)}") 
    
def cnt_face(file_path):
    mesh = open(file_path,'r').read()
    cnt = 0
    for v in mesh.split('\n'):
        if v.startswith('f '):
            cnt += 1
    return cnt

def get_mesh(file_path): 
    mesh = trimesh.load(file_path, force='mesh')
     
    vertices = mesh.vertices.tolist()
    faces = mesh.faces.tolist()
    centered_vertices = vertices - np.mean(vertices, axis=0)
 
    max_abs = np.max(np.abs(centered_vertices))
    vertices = centered_vertices / (max_abs / 0.95)   
      
    def sort_vertices(vertex):
        return vertex[1], vertex[0], vertex[2]   
 
    seen = OrderedDict()
    for point in vertices: 
      key = tuple(point)
      if key not in seen:
        seen[key] = point
        
    unique_vertices =  list(seen.values()) 
    sorted_vertices = sorted(unique_vertices, key=sort_vertices)
     
    vertices_as_tuples = [tuple(v) for v in vertices]
    sorted_vertices_as_tuples = [tuple(v) for v in sorted_vertices]

    vertex_map = {old_index: new_index for old_index, vertex_tuple in enumerate(vertices_as_tuples) for new_index, sorted_vertex_tuple in enumerate(sorted_vertices_as_tuples) if vertex_tuple == sorted_vertex_tuple}
 

    reindexed_faces = [[vertex_map[face[0]], vertex_map[face[1]], vertex_map[face[2]]] for face in faces] 
    return np.array(sorted_vertices), np.array(reindexed_faces)

obj_data = []
base_path = "/f_ndata/zekai/ShapeNetCore.v2/03001627_processed"
cnt = 0
for file in tqdm(sorted(os.listdir(base_path))[:1000]):
    path = os.path.join(base_path, file)
    if cnt_face(path) >= 800:
        continue
    else:
        cnt += 1
    vertices, faces = get_mesh(path)
    obj_data.append({"vertices": torch.tensor(vertices.tolist(), dtype=torch.float).to("cuda"), "faces":  torch.tensor(faces.tolist(), dtype=torch.long).to("cuda"),"texts": "chair"})
    if cnt >= 50:
        break

def convert_face_to_tensor(vertices, faces):
    # aggregate vertices into faces
    # for example:
    # faces = [[0, 1, 2], [1, 2, 3]]
    # vertices = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]
    # faces_coordinates = [[[0, 0, 0], [1, 1, 1], [2, 2, 2]], [[1, 1, 1], [2, 2, 2], [3, 3, 3]]]
    faces_coordinates = vertices[faces]
    


dataset = MeshDataset(obj_data)
dataset.generate_face_edges()
dataset.data = dataset.data*100

max_length =  max(len(d["faces"]) for d in dataset if "faces" in d) 
print(max_length)

autoencoder = MeshAutoencoder(
    dim = 576,
    encoder_depth = 6,
    decoder_depth = 6,
    num_discrete_coors = 128
)

autoencoder_trainer = MeshAutoencoderTrainer(model=autoencoder,learning_rate = 1e-2, 
                                             warmup_steps = 10,
                                             dataset = dataset, 
                                             checkpoint_every_epoch = 20,  
                                             num_train_steps=100,
                                             batch_size=8,
                                             grad_accum_every=1) 
loss = autoencoder_trainer.train(10 ,stop_at_loss = 1.0)   

autoencoder_trainer = MeshAutoencoderTrainer(model=autoencoder,learning_rate = 5e-3, 
                                             warmup_steps = 10,
                                             dataset = dataset, 
                                             checkpoint_every_epoch = 20,  
                                             num_train_steps=100,
                                             batch_size=16,
                                             grad_accum_every=2) 
loss = autoencoder_trainer.train(10 ,stop_at_loss = 0.35)   

max_length =  max(len(d["faces"]) for d in dataset if "faces" in d) 
max_seq =  max_length * 6   
transformer = MeshTransformer(
    autoencoder,
    dim = 512,
    coarse_pre_gateloop_depth = 6,
    fine_pre_gateloop_depth= 4, 
    max_seq_len = max_seq, 
)
 
# dataset.generate_codes(autoencoder)
# test_autoencoder
# codes: [B,F*3,2]

###
self.autoencoder.eval()
face_coords, face_mask = self.autoencoder.decode_from_codes_to_faces(codes)

face_coords = [face_coords_to_file(coords[mask]) for coords, mask in zip(face_coords, face_mask)]


###

trainer = MeshTransformerTrainer(model = transformer,warmup_steps = 10,grad_accum_every=1,num_train_steps=100, dataset = dataset, learning_rate = 1e-1, batch_size=8 , checkpoint_every_epoch=5)
loss = trainer.train(10 ,stop_at_loss = 4e-3) 

trainer = MeshTransformerTrainer(model = transformer,warmup_steps = 10,grad_accum_every=1,num_train_steps=100, dataset = dataset, learning_rate = 5e-3, batch_size=8 , checkpoint_every_epoch=5)
loss = trainer.train(15 ,stop_at_loss = 2e-3) 

trainer = MeshTransformerTrainer(model = transformer,warmup_steps = 10,grad_accum_every=1,num_train_steps=100, dataset = dataset, learning_rate = 1e-3, batch_size=8 , checkpoint_every_epoch=5)
loss = trainer.train(10 ,stop_at_loss = 1e-3) 

coords = []
for r in np.arange(0, 1.0, 0.2):
    print(r)
    faces_coordinates = transformer.generate(temperature=r) 
    coords.append(faces_coordinates) 

all_vertices = []
all_faces = []
vertex_offset = 0

# Translation distance for each model
translation_distance = 1.0  # Adjust as needed

for r, faces_coordinates in enumerate(coords): 
    tensor_data = faces_coordinates[0].cpu()

    numpy_data = tensor_data.numpy().reshape(-1, 3)

    # Translate the model to avoid overlapping
    numpy_data[:, 0] += translation_distance * (r / 0.2 - 1)  # Adjust X coordinate

    # Accumulate vertices
    for vertex in numpy_data:
        all_vertices.append(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

    # Accumulate faces with adjusted indices
    for i in range(1, len(numpy_data), 3):
        all_faces.append(f"f {i + vertex_offset} {i + 1 + vertex_offset} {i + 2 + vertex_offset}\n")

    # Update the vertex offset for the next model
    vertex_offset += len(numpy_data)

# Combine vertices and faces
obj_file_content = "".join(all_vertices) + "".join(all_faces)

# Save to a single file
obj_file_path = f"generated2.obj"
with open(obj_file_path, "w") as file:
    file.write(obj_file_content)