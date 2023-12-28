import torch
import trimesh
import numpy as np
import os
import csv 
import json
import math 
from collections import OrderedDict

from meshgpt_pytorch import (
    MeshTransformerTrainer,
    MeshAutoencoderTrainer,
    MeshAutoencoder,
    MeshTransformer
)

def get_3d_data(file_path): 
    mesh = trimesh.load(file_path, force='mesh')
    
    # Extract vertices and faces
    vertices = mesh.vertices.tolist()
    faces = mesh.faces.tolist()
    centered_vertices = vertices - np.mean(vertices, axis=0)
 
    max_abs = np.max(np.abs(centered_vertices))
    vertices = centered_vertices / (max_abs / 0.95)  
      
      
    def sort_vertices(vertex): # Sort by Y , X, Z.  Y is vertical
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

def augment_mesh_scalar(vertices, scale_factor):
    # Apply a scalar factor to XYZ coordinates
    transformed_vertices = vertices * scale_factor
    return transformed_vertices

def generate_scale_factors(num_examples, lower_limit=0.75, upper_limit=1.25): 
    scale_factors = np.random.uniform(lower_limit, upper_limit, size=num_examples)
    return scale_factors

def jitter_mesh(vertices, jitter_factor=0.01): 
    offsets = np.random.uniform(-jitter_factor, jitter_factor, size=vertices.shape)
 
    jittered_vertices = vertices + offsets 
    return jittered_vertices 

def augment_mesh(vertices, scale_factor):
    #vertices = jitter_mesh(vertices)
    transformed_vertices = vertices * scale_factor
    
    return transformed_vertices
 

def load_models(directory, num_examples, variations):
    obj_datas = []  
    
    print(f"num_examples: {num_examples}")
    for filename in os.listdir(directory):  
        if (filename.endswith(".obj") or  filename.endswith(".glb") or  filename.endswith(".off")):
            file_path = os.path.join(directory, filename)

            scale_factors = generate_scale_factors(variations, 0.7, 0.9) 
            vertices, faces = get_3d_data(file_path) 

            for scale_factor in scale_factors: 
                aug_vertices = augment_mesh(vertices.copy(), scale_factor) 
                
                for _ in range(num_examples):
                    obj_data = {"vertices": aug_vertices.tolist(), "faces":  faces.tolist(), "texts": filename[:-4]}
                    obj_datas.append(obj_data)   
    return obj_datas
  


  
def load_json(file,num_examples):
    obj_datas = []
    with open(file, "r") as json_file:
        loaded_data = json.load(json_file) 
        for item in loaded_data:
            for _ in range(num_examples): 
                obj_data = {"vertices": torch.tensor(item["vertices"], dtype=torch.float).to("cuda"), "faces":  torch.tensor(item["faces"], dtype=torch.long).to("cuda"),"texts": item["texts"] } 
                obj_datas.append(obj_data)
    return obj_datas
                        
         
import torch
from torch import Tensor, tensor
from torch.utils.data import Dataset, DataLoader 
from tqdm import tqdm
import numpy as np 
import gc
from torch.nn.utils.rnn import pad_sequence
from meshgpt_pytorch.data import ( 
    derive_face_edges_from_faces
) 
 
class MeshDataset(Dataset): 
    
    def __init__(self, data): 
        self.data = data
        print(f"Got {len(data)} data")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        data = self.data[idx] 
        return data  
    
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

import json
#tables = load_models(r" filtered",5,5)  
#with open("data.json", "w") as json_file:
#    json.dump(tables, json_file) 

tables = load_json("/f_ndata/zekai/ShapeNetCore.v2/table100.json",2)
dataset = MeshDataset(tables) 
dataset.generate_face_edges()
dataset.data[0].keys()

desired_order = ['vertices', 'faces', 'face_edges', 'texts']

dataset.data = [
    {key: d[key] for key in desired_order} for d in dataset.data
]

unique_values = set(item["texts"] for item in dataset.data)

print(len(unique_values))  
print(unique_values)

autoencoder = MeshAutoencoder( 
    dim = 512,
    use_residual_lfq = True,
    commit_loss_weight = 0.1,
    bin_smooth_blur_sigma = 0.4
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
                                             batch_size=2,
                                             grad_accum_every=1)

loss = autoencoder_trainer.train(40,stop_at_loss = 0.25)   
autoencoder_trainer = MeshAutoencoderTrainer(model =autoencoder,learning_rate = 1e-4, 
                                             warmup_steps = 10,
                                             dataset = dataset,
                                             checkpoint_every_epoch = 20,  
                                             num_train_steps=100,
                                             batch_size=2,
                                             grad_accum_every=1)

loss = autoencoder_trainer.train(180,stop_at_loss = 0.25)   
autoencoder_trainer.save(f'./mesh-encoder_2_loss_{loss:.3f}.pt') 

max_length =  max(len(d["faces"]) for d in dataset if "faces" in d) 
max_seq =  max_length * 6  
print(max_length)
print(max_seq)
transformer = MeshTransformer(
    autoencoder,
    dim = 512,
    max_seq_len = max_seq,
    coarse_pre_gateloop_depth = 6,
    fine_pre_gateloop_depth= 4, 
    condition_on_text = False
)
total_params = sum(p.numel() for p in transformer.parameters())
print(f"Total parameters: {total_params}") 

 
trainer = MeshTransformerTrainer(model = transformer,warmup_steps = 10,grad_accum_every=1,num_train_steps=100,   dataset = dataset,
                                 learning_rate = 1e-1, batch_size=1)
trainer.train(80,stop_at_loss = 0.00009)   

 
trainer = MeshTransformerTrainer(model = transformer,warmup_steps = 10,grad_accum_every=1,num_train_steps=100,  dataset = dataset,
                                 learning_rate = 1e-2, batch_size=1)
trainer.train(80,stop_at_loss = 0.00009)    

trainer = MeshTransformerTrainer(model = transformer,warmup_steps = 10,grad_accum_every=1,num_train_steps=100,  dataset = dataset,
                                 learning_rate = 1e-4, batch_size=1)
loss = trainer.train(80,stop_at_loss = 0.00009)   

trainer.save(f'./mesh-transformer_2_{loss:.3f}.pt')    


unique_values = set(item["texts"] for item in dataset.data)
print(len(unique_values))  
coords = []
for text in unique_values: 
    print(f"doing {text}")
    faces_coordinates = transformer.generate(texts = [text]) 
    coords.append(faces_coordinates)
    tensor_data = faces_coordinates[0].cpu()
    
    numpy_data = tensor_data.numpy().reshape(-1, 3)
    
    obj_file_content = ""
    
    for vertex in numpy_data:
        obj_file_content += f"v {vertex[0]} {vertex[1]} {vertex[2]}\n"

    for i in range(1, len(numpy_data), 3):
        obj_file_content += f"f {i} {i + 1} {i + 2}\n"

    # Save to a file
    obj_file_path = f'./tests/3d_output_{text}.obj'
    with open(obj_file_path, "w") as file:
        file.write(obj_file_content)

    print(obj_file_path) 
    
    
all_vertices = []
all_faces = []
vertex_offset = 0
 
translation_distance = 0.3  

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
obj_file_path = f"./tests/3d_models_all.obj"
with open(obj_file_path, "w") as file:
    file.write(obj_file_content)

print(obj_file_path)


coords_all = []
for text in set(item["texts"] for item in dataset.data): 
    print(f"Doing {text}")
    coords = []
    for r in np.arange(0, 1.0, 0.1):
        faces_coordinates = transformer.generate(temperature=r, texts = [text]) 
        coords.append(faces_coordinates)
    coords_all.append(coords)
    
    all_vertices = []
    all_faces = []
    vertex_offset = 0

    # Translation distance for each model
    translation_distance = 0.3  # Adjust as needed

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
    obj_file_path = f"./results/3d_models_{text}_temps.obj"
    with open(obj_file_path, "w") as file:
        file.write(obj_file_content)

    print(obj_file_path)


def loadModels():
    autoencoder = MeshAutoencoder(
        dim = 576,
        encoder_depth = 6,
        decoder_depth = 6,
        num_discrete_coors = 128  ,
        local_attn_depth =0, 
        
    )
    autoencoder_trainer = MeshAutoencoderTrainer(model = autoencoder,
                                    learning_rate = 1e-1, 
                                                checkpoint_every_epoch= 5,
                                                warmup_steps = 10,
                                                dataset = dataset,  
                                                num_train_steps=100,
                                                batch_size=2,
                                                grad_accum_every=1)

    autoencoder_trainer.load(r"mesh-encoder_last.pt")
    encoder = autoencoder_trainer.model
    max_length =  max(len(d["faces"]) for d in dataset if "faces" in d) 
    max_seq =  max_length * 6  
    
    transformer = MeshTransformer(
        autoencoder,
        dim = 768,
        max_seq_len = max_seq,
        condition_on_text = True)
     
    trainer = MeshTransformerTrainer(model = transformer,warmup_steps = 10,grad_accum_every=1,num_train_steps=100, checkpoint_folder = r"F:\MachineLearning\Mesh\MeshGPT\checkpoints" , dataset = dataset,
                                    learning_rate = 1e-3, batch_size=2) 
    trainer.load(r"mesh-transformer.pt")
    transformer = trainer.model
    return transformer, encoder

#transformer, autoencoder =  loadModels() 