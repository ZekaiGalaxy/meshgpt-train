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

np.random.seed(42)

def check_duplicate_vertices(vertices):
    unique_vertices, counts = np.unique(vertices, axis=0, return_counts=True)
    duplicates = unique_vertices[counts > 1]
    return len(duplicates) > 0

def get_3d_data(file_path): 
    mesh = trimesh.load(file_path, force='mesh')
    vertices = mesh.vertices
    faces = mesh.faces   
    # centered
    centered_vertices = vertices - np.mean(vertices, axis=0)
    # normalize to unit length
    max_abs = np.max(np.abs(centered_vertices))
    vertices_normalized = centered_vertices / (max_abs / 0.95)  
    # sort vertices: y, x, z
    vertices_sorted_indices = np.lexsort((vertices_normalized[:, 1], vertices_normalized[:, 0], vertices_normalized[:, 2]))
    vertices_normalized_sorted = vertices_normalized[vertices_sorted_indices]
    # Convert indices to tuples for creating Look-Up Table (LUT)
    tuples_sorted_indices = [tuple([index]) for index in vertices_sorted_indices.tolist()]
    # Create Look-Up Table (LUT)
    lut = {old_index[0]: new_index for new_index, old_index in enumerate(tuples_sorted_indices)}
    # Reindex faces using LUT
    faces_reindexed = np.vectorize(lut.get, otypes=[int])(faces) 
    # sort faces based on their lowest vertex index
    faces_sorted = faces_reindexed[np.lexsort(faces_reindexed.T)]
    print(f"{file_path} vertices {len(vertices)} faces {len(faces)}")
    return vertices_normalized_sorted, faces_sorted 

def augment_mesh_scalar(vertices, scale_factor):
    transformed_vertices = vertices * scale_factor
    return transformed_vertices

def resize_mesh_to_unit_length(vertices):
    min_vals = np.min(vertices, axis=0)
    max_vals = np.max(vertices, axis=0)
    ranges = max_vals - min_vals
    longest_side = np.max(ranges)
    scaled_vertices = vertices / longest_side
    return scaled_vertices

def generate_scale_factors(num_examples, lower_limit=0.75, upper_limit=1.25): 
    scale_factors = np.random.uniform(lower_limit, upper_limit, size=(num_examples,3))
    return scale_factors

def jitter_mesh(vertices, jitter_factor=0.01): 
    offsets = np.random.uniform(-jitter_factor, jitter_factor, size=vertices.shape)
    jittered_vertices = vertices + offsets 
    return jittered_vertices 

def augment_mesh(vertices, scale_factor):
    vertices = jitter_mesh(vertices)
    transformed_vertices = vertices * scale_factor
    transformed_vertices = resize_mesh_to_unit_length(transformed_vertices)
    return transformed_vertices

def load_models(directory, variations):
    obj_datas = []  
    for checklist in tqdm(os.listdir(directory)):  
        for filename in os.listdir(os.path.join(directory, checklist,'models')):
            if (filename.endswith(".obj") or  filename.endswith(".glb") or  filename.endswith(".off")):
                file_path = os.path.join(directory, checklist,'models',filename)

                scale_factors = generate_scale_factors(variations, 0.75, 1.25) 
                vertices, faces = get_3d_data(file_path) 
                if len(faces) > 800:
                    # print('discard faces over 800...')
                    continue
                obj_data = {"vertices": vertices.tolist(), "faces":  faces.tolist(), "texts": ""}
                obj_datas.append(obj_data)  

                for scale_factor in scale_factors: 
                    aug_vertices = augment_mesh(vertices.copy(), scale_factor) 
                    obj_data = {"vertices": aug_vertices.tolist(), "faces":  faces.tolist(), "texts": ""}
                    obj_datas.append(obj_data) 
      
    return obj_datas

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
# table
tables = load_models("/f_ndata/zekai/ShapeNetCore.v2/04379243",4)  
print(len(tables))
# for x in tables[:10]:
#     print(x)
with open("/f_ndata/zekai/ShapeNetCore.v2/table.json", "w") as json_file:
   json.dump(tables, json_file)


# tables = load_json("/kaggle/input/shapenet/data.json",2)
# dataset = MeshDataset(tables) 

# class MeshDataset(Dataset): 
    
#     def __init__(self, data): 
#         self.data = data
#         print(f"Got {len(data)} data")

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx): 
#         return self.data[idx]
    
#     def embed_texts(self,transformer): 
#         unique_texts = set(item['texts'] for item in self.data)
 
#         text_embeddings = transformer.embed_texts(list(unique_texts))
#         print(f"Got text_embeddings: {len(text_embeddings)}") 
#         text_embedding_dict = dict(zip(unique_texts, text_embeddings))
 
#         for item in self.data:
#             text_value = item['texts']
#             item['text_embeds'] = text_embedding_dict.get(text_value, None)
#             del item['texts']
 
        
#     def sample_obj(self):
#         all_vertices = []
#         all_faces = []
#         vertex_offset = 0 


#         translation_distance = 0.5  # Adjust as needed 
#         vertex_offset = len(all_vertices)
        
#         for r, faces_coordinates in enumerate(self.data):    
#             if r > 30:
#                 break
#             for vertex in faces_coordinates["vertices"]: 
#                 all_vertices.append(f"v {vertex[0]+translation_distance * (r / 0.2 - 1)} {vertex[1]} {vertex[2]}\n")
#                 #all_vertices.append(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

#             for face in faces_coordinates["faces"]:
#                 all_faces.append(f"f {face[0]+1+vertex_offset} {face[1]+1+vertex_offset} {face[2]+1+vertex_offset}\n") 
#                 try:
#                     all_vertices[face[0]+vertex_offset]
#                     all_vertices[face[1]+vertex_offset]
#                     all_vertices[face[2]+vertex_offset]
#                 except Exception  as e :
#                     print(e)
#                     print(face[0]+vertex_offset)
#                     print(face[1]+vertex_offset)
#                     print(face[2]+vertex_offset)
#                     print(len(all_vertices))
                
#             vertex_offset = len(all_vertices) 


#         obj_file_content = "".join(all_vertices) + "".join(all_faces)

#         # Save to a single file
#         obj_file_path = "./combined_3d_models.obj"
#         with open(obj_file_path, "w") as file:
#             file.write(obj_file_content)

#         print(obj_file_path)