import torch

from meshgpt_pytorch import (
    MeshAutoencoder,
    MeshTransformer,
    MeshAutoencoderTrainer,
    MeshTransformerTrainer,
    DatasetFromTransforms
)

# autoencoder

autoencoder = MeshAutoencoder(
    num_discrete_coors = 128,
    local_attn_encoder_depth = 0,
    local_attn_decoder_depth = 2
)

# mock dataset

from torch.utils.data import Dataset

class MockDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        from random import randrange
        return torch.randn(randrange(20, 40), 3), torch.randint(0, 10, (randrange(20, 40), 3))

dataset = MockDataset()

trainer = MeshAutoencoderTrainer(
    autoencoder,
    dataset = dataset,
    val_dataset = dataset,
    batch_size = 2,
    grad_accum_every = 2,
    val_every = 1,
    num_train_steps = 1,
    checkpoint_every = 5,
    accelerator_kwargs = dict(
        cpu = True
    )
)

trainer.save('./autoencoder.pt')
trainer.load('./autoencoder.pt')
trainer()


transformer_trainer.train(run.config.transformer_train)

codes, continuous_coors = transformer.generate(return_codes=True)

codes_list = codes.cpu().tolist()

import json

with open("output_codes.json", "w") as f:
    json.dump(codes_list, f)

continuous_coors_list = continuous_coors.cpu().tolist()

with open("continuous_coors.json", "w") as f:
    json.dump(continuous_coors.tolist(), f)

flat_list = [item for sublist in continuous_coors_list for item in sublist]

vertices = [vertex for sublist in flat_list for vertex in sublist]

faces = [[i, i + 1, i + 2] for i in range(0, len(vertices), 3)]


I trained for 10 epochs (200 examples dataset) but seems like 3-5 epoch would work as well. I used 1e-3 lr for encoder and 1e-2 lr for transformer since it's training only on one shape

it's only the same chair x 200 times in the dataset

output = transformer.generate()
print(output .shape)
print(output)

torch.Size([1, 20, 3, 3])
tensor([[[[-0.8047,  0.1953,  0.8516],
          [-0.3359,  0.8203,  0.9609],
          [ 0.3672, -0.5859,  0.0078]],

         [[-0.4922, -0.7578, -0.3828],
          [-0.2422,  0.2266,  0.2578],
          [-0.2578, -0.8516, -0.3516]],


import torch
from torch.utils.data import Dataset
import os
import json
import trimesh
import numpy as np
import sys
import functools
import wandb

from abc import abstractmethod
import os
import random
from scipy.spatial.transform import Rotation as R


class MeshDataset(Dataset):
    def __init__(self, folder_path, augments_per_item):
        self.folder_path = folder_path
        self.file_list = os.listdir(folder_path)
        self.supported_formats = (".glb", ".gltf")
        self.augments_per_item = augments_per_item
        self.seed = 42

    def get_max_face_count(self):
        max_faces = 0
        files = self.filter_files()
        files = sorted(self.filter_files())
        for file in files:
            file_path = os.path.join(self.folder_path, file)
            scene = trimesh.load(file_path, force="scene")
            total_faces_in_file = 0
            for _, geometry in scene.geometry.items():
                try:
                    geometry.apply_transform(scene.graph.get(_)[0])
                except Exception as e:
                    pass

                num_faces = len(geometry.faces)
                total_faces_in_file += num_faces

            if total_faces_in_file > max_faces:
                max_faces = total_faces_in_file

        return max_faces

    def filter_files(self):
        filtered_list = [
            file for file in self.file_list if file.endswith(self.supported_formats)
        ]
        return filtered_list

    @staticmethod
    def convert_to_glb(json_data, output_file_path):
        scene = trimesh.Scene()
        vertices = np.array(json_data[0])
        faces = np.array(json_data[1])
        if faces.max() >= len(vertices):
            raise ValueError(
                f"Face index {faces.max()} exceeds number of vertices {len(vertices)}"
            )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        scene.add_geometry(mesh)
        with open(output_file_path, "wb") as f:
            f.write(scene.export(file_type="glb"))

    @staticmethod
    def convert_to_obj(json_data, output_file_path):
        scene = trimesh.Scene()
        vertices = np.array(json_data[0])
        faces = np.array(json_data[1])
        if faces.max() >= len(vertices):
            raise ValueError(
                f"Face index {faces.max()} exceeds number of vertices {len(vertices)}"
            )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        scene.add_geometry(mesh)
        with open(output_file_path, "w") as f:
            f.write(scene.export(file_type="obj"))

    @staticmethod
    def compare_json(json_data1, json_data2):
        if len(json_data1) != len(json_data2):
            return False

        if not np.array_equal(json_data1[0], json_data2[0]):
            return False

        if not np.array_equal(json_data1[1], json_data2[1]):
            return False

        return True

    def __len__(self):
        return len(self.filter_files()) * self.augments_per_item

    @staticmethod
    def compare_vertices(vertex_a, vertex_b):
        # glTF uses right-handed coordinate system (Y-Z-X).
        # Y is up and is different from the meshgpt paper.
        for i in [1, 2, 0]:  # Compare Y, then Z, then X
            if vertex_a[i] < vertex_b[i]:
                return -1
            elif vertex_a[i] > vertex_b[i]:
                return 1
        return 0  # If all coordinates are equal

    @staticmethod
    def compare_faces(face_a, face_b, vertices):
        for i in range(3):
            # Check if face indices are within the range of vertices list
            if face_a[i] >= len(vertices) or face_b[i] >= len(vertices):
                raise IndexError("Face index out of range")

            vertex_comparison = MeshDataset.compare_vertices(
                vertices[face_a[i]], vertices[face_b[i]]
            )
            if vertex_comparison != 0:
                return vertex_comparison

        return 0

    def load_and_process_scene(self, idx):
        files = self.filter_files()
        file_idx = idx // self.augments_per_item
        augment_idx = idx % self.augments_per_item
        file_path = os.path.join(self.folder_path, files[file_idx])

        _, file_extension = os.path.splitext(file_path)

        scene = trimesh.load(file_path, force="scene")

        all_triangles = []
        all_faces = []
        all_vertices = []

        for mesh_idx, (name, geometry) in enumerate(scene.geometry.items()):
            vertex_indices = {}

            try:
                geometry.apply_transform(scene.graph.get(name)[0])
            except Exception as e:
                pass

            vertices = geometry.vertices
            vertices = [tuple(v) for v in vertices]
            vertex_indices.update({v: i for i, v in enumerate(vertices)})

            geometry.vertices = np.array(vertices)

            offset = len(all_vertices)

            faces = [
                [
                    vertex_indices[tuple(geometry.vertices[vertex])] + offset
                    for vertex in face
                ]
                for face in geometry.faces
            ]

            faces = [[vertex for vertex in face] for face in faces]

            all_faces.extend(faces)
            all_vertices.extend(vertices)

        all_faces.sort(
            key=functools.cmp_to_key(
                lambda a, b: MeshDataset.compare_faces(a, b, all_vertices)
            )
        )

        return all_faces, all_vertices, augment_idx

    def create_new_vertices_and_faces(self, all_faces, all_vertices):
        new_vertices = []
        new_faces = []
        vertex_map = {}

        import math

        def calculate_angle(point, center):
            return math.atan2(point[1] - center[1], point[0] - center[0])

        def sort_vertices_ccw(vertices):
            center = [
                sum(vertex[i] for vertex in vertices) / len(vertices) for i in range(2)
            ]

            return sorted(vertices, key=lambda vertex: -calculate_angle(vertex, center))

        def calculate_normal(face_vertices):
            v1 = np.array(face_vertices[1]) - np.array(face_vertices[0])
            v2 = np.array(face_vertices[2]) - np.array(face_vertices[0])
            return np.cross(v1, v2)

        for face in all_faces:
            new_face = []
            for vertex_index in face:
                if vertex_index not in vertex_map:
                    new_vertex = all_vertices[vertex_index]
                    new_vertices.append(new_vertex)
                    vertex_map[vertex_index] = len(new_vertices) - 1
                new_face.append(vertex_map[vertex_index])

            new_face_vertices = [new_vertices[i] for i in new_face]

            original_normal = calculate_normal(new_face_vertices)

            sorted_vertices = sort_vertices_ccw(new_face_vertices)

            new_normal = calculate_normal(sorted_vertices)

            if np.dot(original_normal, new_normal) < 0:
                sorted_vertices = list(reversed(sorted_vertices))

            sorted_indices = [
                new_face[new_face_vertices.index(vertex)] for vertex in sorted_vertices
            ]

            new_faces.append(sorted_indices)

        return new_vertices, new_faces

    def __getitem__(self, idx):
        all_faces, all_vertices, augment_idx = self.load_and_process_scene(idx)
        new_vertices, new_faces = self.create_new_vertices_and_faces(
            all_faces, all_vertices
        )

        return self.augment_mesh(
            (
                torch.tensor(new_vertices, dtype=torch.float),
                torch.tensor(new_faces, dtype=torch.long),
            ),
            self.augments_per_item,
            augment_idx,
        )

    def augment_mesh(self, base_mesh, augment_count, augment_idx):
        # Set the random seed for reproducibility
        random.seed(self.seed + augment_count * augment_idx + augment_idx)

        # Generate a random scale factor
        scale = random.uniform(0.8, 1)

        vertices = base_mesh[0]

        # Calculate the centroid of the object
        centroid = [
            sum(vertex[i] for vertex in vertices) / len(vertices) for i in range(3)
        ]

        # Translate the vertices so that the centroid is at the origin
        translated_vertices = [[v[i] - centroid[i] for i in range(3)] for v in vertices]

        # Scale the translated vertices
        scaled_vertices = [
            [v[i] * scale for i in range(3)] for v in translated_vertices
        ]

        # Jitter the vertices
        jittered_vertices = [
            [v[i] + random.uniform(-1.0/256.0, 1.0/256.0) for i in range(3)]
            for v in scaled_vertices
        ]

        # Translate the vertices back so that the centroid is at its original position
        final_vertices = [
            [v[i] + centroid[i] for i in range(3)] for v in jittered_vertices
        ]

        # Normalize uniformly to fill [-1, 1]
        min_vals = np.min(final_vertices, axis=0)
        max_vals = np.max(final_vertices, axis=0)

        # Calculate the maximum absolute value among all vertices
        max_abs_val = max(np.max(np.abs(min_vals)), np.max(np.abs(max_vals)))

        # Calculate the scale factor as the reciprocal of the maximum absolute value
        scale_factor = 1 / max_abs_val if max_abs_val != 0 else 1

        # Apply the normalization
        final_vertices = [
            [(component - c) * scale_factor for component, c in zip(v, centroid)]
            for v in final_vertices
        ]

        return (
            torch.from_numpy(np.array(final_vertices, dtype=np.float32)),
            base_mesh[1],
        )


import unittest
import json


class TestMeshDataset(unittest.TestCase):
    def setUp(self):
        self.augments = 3
        self.dataset = MeshDataset("unit_test", self.augments)

    def test_mesh_augmentation(self):
        for i in range(self.augments):
            mesh = [tensor.tolist() for tensor in self.dataset.__getitem__(i)]
            with open(f"unit_augment/mesh_{str(i).zfill(2)}.json", "wb") as f:
                f.write(json.dumps(mesh).encode())
            self.dataset.convert_to_glb(
                mesh, f"unit_augment/mesh_{str(i).zfill(2)}.glb"
            )


if __name__ == "__main__":
    unittest.main()


When I have been successful, the encoder loss was less 0.200- 0.250 and the loss for the transformer was around 0.00007.

How many steps are that at? I require about 2000 steps since 200 x10 epochs = 2000.

How many examples/steps of the same 3d mesh did you train it on? I trained for 10-20 epochs @ 2000 examples and got 0.19 loss.
I think you are training on too few examples, it needs massive amounts of data to model. And if you do data augmentation you'll need even more data, maybe 30-40 epochs or more.

I might have worded that badly but no, I'm using the same model without any augmentations.
But train for 10/20 epochs @ 2000 items per dataset and let me know.
Kaggle has some awesome free GPU's.

Also, I'm current training on like 6 3d mesh chairs. Each chair has 1500 examples, but it have 3 augmentation version .
So each 3d mesh file have a total of 500 x 3 =1500 examples.

The total is 12 000 examples.

The learning rate seems bit high, for the encoder i used 1e-3 (0.001) and for the transformer i used 1e-2 (0.01).
When the loss becomes quite low for the transformer you can try using a lower learning rate such as 1e-3.

I see that the dataset size is 10, for training effective I just duplicate the one model x2000 times since it can train faster I think when dealing with bigger loads.

Using a dataset of 2 chairs with 5000 examples (2 meshes, 5 augmentations x 500)
I got the encoder to 0.2 loss after 2 epochs but the transformer is at 0.001695 loss after 40 epochs and taken 2h's.

