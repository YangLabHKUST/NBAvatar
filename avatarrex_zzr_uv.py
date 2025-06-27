import trimesh
import numpy as np
from utils.obj_io import load_obj_data, save_obj_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp_ply", type=str, required=True)
    parser.add_argument("--inp_uv", type=str, required=True)
    parser.add_argument("--out_obj", type=str, required=True)
    args = parser.parse_args()

    inp_mesh = trimesh.load(args.inp_ply, process=False)
    inp_uv = load_obj_data(args.inp_uv)
    vt = inp_uv['vt']
    ft = inp_uv['ft']

    model = {'v': inp_mesh.vertices, 'f': inp_mesh.faces, 'vt': vt, 'ft': ft}
    save_obj_data(model, args.out_obj)