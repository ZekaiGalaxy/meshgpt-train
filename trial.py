import torch
def convert_face_to_tensor(vertices, faces):
    # aggregate vertices into faces
    # for example:
    # faces = [[0, 1, 2], [1, 2, 3]]
    # vertices = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]
    # faces_coordinates = [[[0, 0, 0], [1, 1, 1], [2, 2, 2]], [[1, 1, 1], [2, 2, 2], [3, 3, 3]]]
    faces_coordinates = vertices[faces]
    return faces_coordinates

vertices = torch.tensor(
    [
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]
    ]
)

faces = torch.tensor(
    [
        [0, 1, 2],
        [1, 2, 3]
    ]
)

print(convert_face_to_tensor(vertices, faces))