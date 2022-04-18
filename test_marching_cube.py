
from binvox_rw import read_as_3d_array
import pytorch3d.ops
import torch
import pytorch3d.io.obj_io

modelName = "plane"
with open('./models/{}.binvox'.format(modelName), 'rb') as fp:
    v = read_as_3d_array(fp)
    voxel = torch.tensor(v.data)
    voxel = voxel.expand(1, *voxel.shape)
    meshes = pytorch3d.ops.cubify(voxel, 0.2)
    print(meshes)
    mesh = meshes[0]
    verts = mesh.verts_list()[0]
    faces = mesh.faces_list()[0]
    pytorch3d.io.obj_io.save_obj("./models/{}.obj".format(modelName), verts, faces)
    print("done")    
