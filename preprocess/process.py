from tqdm import tqdm
import os

def process_obj(input_path, output_path):
    obj_file = open(input_path, 'r').read()
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

    # now print it to a new obj file:
    obj_file = open(output_path, 'w')
    for v in vertices:
        obj_file.write(f"v {v[0]} {v[1]} {v[2]}\n")
    for m in meshes:
        obj_file.write(f"f {m[0]+1} {m[1]+1} {m[2]+1}\n")
    obj_file.close()

# base path:/f_ndata/zekai/ShapeNetCore.v2/03001627 chair
for folder in tqdm(os.listdir("/f_ndata/zekai/ShapeNetCore.v2/03001627")):
    path = os.path.join("/f_ndata/zekai/ShapeNetCore.v2/03001627", folder)
    if os.path.isdir(path):
        input_path = os.path.join(path, "models/model_normalized.obj")
        output_path = "/f_ndata/zekai/ShapeNetCore.v2/03001627_processed/" + folder + ".obj"
        process_obj(input_path, output_path)
        