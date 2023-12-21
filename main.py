import torch
from torch.utils.data import Dataset, DataLoader 
from tqdm import tqdm
import os

def process_obj(obj_file):
    # vertices
    vertices = []
    for v in obj_file.split('\n'):
        if v.startswith('v '):
            x,y,z = v.replace('v ','').strip().split(" ")
            vertices.append([float(x),float(y),float(z)])

    # meshes
    meshes = []
    shapes = obj_file.split('# Mesh ')[1:]
    for shape in shapes:
        mesh = []
        for face in shape.split('\n'):
            if not face.startswith('f '):
                continue
            face_str = face.replace('f ','').strip()
            v1,v2,v3 = face_str.split(" ")
            mesh.append([int(v1.split('/')[0])-1,int(v2.split('/')[0])-1,int(v3.split('/')[0])-1])
        if len(mesh) > 0:
            meshes.extend(mesh)

    return vertices, meshes

# obj_data = {"texts": "chair", "vertices": vertices, "faces": faces} 
obj_data = []
object_path = "/f_ndata/zekai/ShapeNet/ShapeNetCore.v2/03001627"
for checklist in tqdm(os.listdir(object_path)[:500]):
    path = os.path.join(object_path, checklist, "models/model_normalized.obj")
    obj_file = open(path,'r').read()
    vertices, meshes = process_obj(obj_file)
    if len(meshes) > 800:
        continue
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32)
    meshes_tensor = torch.tensor(meshes, dtype=torch.int64)
    print(vertices_tensor.shape)
    print(meshes_tensor.shape)
    obj_data.append({"texts":"chair","vertices": vertices_tensor, "faces": meshes_tensor})
# path = 'data/model_normalized.obj'
# obj_file = open(path,'r').read()
# vertices, meshes = process_obj(obj_file)
# vertices_tensor = torch.tensor(vertices, dtype=torch.float32)
# meshes_tensor = torch.tensor(meshes, dtype=torch.int64)
# print(vertices_tensor.shape)
# print(meshes_tensor.shape)
# obj_data = {"texts":"chair","vertices": vertices_tensor, "faces": meshes_tensor}

class MeshDataset(Dataset): 
    def __init__(self, obj_data): 
        self.obj_data = obj_data
        print(f"Got {len(obj_data)} data")

    def __len__(self):
        return len(self.obj_data)

    def __getitem__(self, idx):
       return  self.obj_data[idx] 

dataset = MeshDataset(obj_data)

from meshgpt_pytorch import (
    MeshAutoencoder,
    MeshTransformer,
    MeshTransformerTrainer,
    MeshAutoencoderTrainer
)

autoencoder = MeshAutoencoder(
    dim = 512,
    encoder_depth = 6,
    decoder_depth = 6,
    num_discrete_coors = 128
)
transformer = MeshTransformer(
    autoencoder,
    dim = 512,
    max_seq_len = 768
)

autoencoder_trainer = MeshAutoencoderTrainer(model = autoencoder,learning_rate = 1e-5, warmup_steps = 10,dataset = dataset,batch_size=16,grad_accum_every=1,num_train_steps=100)
autoencoder_trainer.train(100,True)

# max_length =  max(len(d["faces"]) for d in dataset if "faces" in d)
# max_seq =  max_length * 6
# print(max_length)
# print(max_seq)
# transformer = MeshTransformer(
#     autoencoder,
#     dim = 16,
#     max_seq_len = max_seq,
#     #condition_on_text = True
# )
 
 
# trainer = MeshTransformerTrainer(model = transformer,warmup_steps = 10, dataset = dataset,learning_rate = 1e-2,batch_size=1,grad_accum_every=1,num_train_steps=1)
# trainer.train(10)