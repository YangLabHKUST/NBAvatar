import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import numpy as np
import torch
import torch.nn.functional as F
import cv2 as cv
import trimesh
import yaml
import tqdm
import math
import nvdiffrast.torch as dr

import smplx
from network.volume import CanoBlendWeightVolume
import config
from utils.obj_io import load_obj_data


def save_pos_map(pos_map, path):
    mask = np.linalg.norm(pos_map, axis = -1) > 0.
    positions = pos_map[mask]
    print('Point nums %d' % positions.shape[0])
    pc = trimesh.PointCloud(positions)
    pc.export(path)


def interpolate_lbs(pts, vertices, faces, vertex_lbs):
    from utils.posevocab_custom_ops.nearest_face import nearest_face_pytorch3d
    from utils.geo_util import barycentric_interpolate
    dists, indices, bc_coords = nearest_face_pytorch3d(
        torch.from_numpy(pts).to(torch.float32).cuda()[None],
        torch.from_numpy(vertices).to(torch.float32).cuda()[None],
        torch.from_numpy(faces).to(torch.int64).cuda()
    )
    # print(dists.mean())
    lbs = barycentric_interpolate(
        vert_attris = vertex_lbs[None].to(torch.float32).cuda(),
        faces = torch.from_numpy(faces).to(torch.int64).cuda()[None],
        face_ids = indices,
        bc_coords = bc_coords
    )
    return lbs[0].cpu().numpy()


map_size = 1024


if __name__ == '__main__':
    from argparse import ArgumentParser
    import importlib

    arg_parser = ArgumentParser()
    arg_parser.add_argument('-c', '--config_path', type = str, help = 'Configuration file path.')
    arg_parser.add_argument('--frame_range', type=int, nargs=3, default = None, help = 'Frame range.')
    args = arg_parser.parse_args()

    opt = yaml.load(open(args.config_path, encoding = 'UTF-8'), Loader = yaml.FullLoader)
    if args.frame_range is not None:
        opt['train']['data']['frame_range'] = list(args.frame_range)
        opt['test']['data']['frame_range'] = list(args.frame_range)
    dataset_module = opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
    MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)
    dataset = MvRgbDataset(**opt['train']['data'])
    data_dir, frame_list = dataset.data_dir, dataset.pose_list

    os.makedirs(data_dir + '/smpl_pos_map', exist_ok = True)
    cano_renderer = dr.RasterizeCudaContext()

    smpl_model = smplx.SMPLX(config.PROJ_DIR + '/smpl_files/smplx', gender = 'neutral', use_pca = False, num_pca_comps = 45, flat_hand_mean = True, batch_size = 1)
    smpl_data = np.load(data_dir + '/smpl_params.npz')
    smpl_data = {k: torch.from_numpy(v.astype(np.float32)) for k, v in smpl_data.items()}

    with torch.no_grad():
        cano_smpl = smpl_model.forward(
            betas = smpl_data['betas'],
            global_orient = config.cano_smpl_global_orient[None],
            transl = config.cano_smpl_transl[None],
            body_pose = config.cano_smpl_body_pose[None]
        )
        cano_smpl_v = cano_smpl.vertices[0].cpu().numpy()
        cano_center = 0.5 * (cano_smpl_v.min(0) + cano_smpl_v.max(0))
        # cano_smpl_v_min = cano_smpl_v.min()
        smpl_faces = smpl_model.faces.astype(np.int64)

    if os.path.exists(data_dir + '/template.obj'):
        print('# Loading template from %s' % (data_dir + '/template.obj'))
        template = load_obj_data(data_dir + '/template.obj')
        using_template = True
    else:
        print(f'# Cannot find template.ply from {data_dir}, using SMPL-X as template')
        import ipdb; ipdb.set_trace()
        template = {'v': cano_smpl_v, 'f': smpl_faces, 'vt': None, 'ft': None}
        using_template = False

    cano_smpl_v = template['v'].astype(np.float32)
    smpl_faces = template['f'].astype(np.int32)
    vt = template['vt'].astype(np.float32)
    ft = template['ft'].astype(np.int32)
    mesh = trimesh.Trimesh(cano_smpl_v, smpl_faces, process = False)
    cano_smpl_n = mesh.vertex_normals.astype(np.float32)

    v = torch.from_numpy(cano_smpl_v).cuda()
    n = torch.from_numpy(cano_smpl_n).cuda()
    f = torch.from_numpy(smpl_faces).cuda()
    vt = torch.from_numpy(vt).cuda()
    ft = torch.from_numpy(ft).cuda()

    vt_ndc = vt * 2 - 1
    vt_ndc[..., -1] *= -1
    vt_ndc = F.pad(vt_ndc, (0, 1), 'constant', 0.)
    vt_ndc = F.pad(vt_ndc, (0, 1), 'constant', 1.)
    rast, rast_db = dr.rasterize(cano_renderer, vt_ndc[None], ft, [map_size, map_size], grad_db=False)

    # render canonical smpl position maps
    out, _ = dr.interpolate(v.contiguous(), rast, f, rast_db=rast_db)
    cano_pos_map = out[0].cpu().numpy()
    cv.imwrite(data_dir + '/smpl_pos_map/cano_smpl_pos_map.exr', cano_pos_map)
    
    # render canonical smpl normal maps
    out, _ = dr.interpolate(n.contiguous(), rast, f, rast_db=rast_db)
    cano_nml_map = out[0].cpu().numpy()
    cv.imwrite(data_dir + '/smpl_pos_map/cano_smpl_nml_map.exr', cano_nml_map)

    body_mask = np.linalg.norm(cano_pos_map, axis = -1) > 0.
    print('number of valid points: %d' % np.sum(body_mask))
    cano_pts = cano_pos_map[body_mask]
    if using_template:
        weight_volume = CanoBlendWeightVolume(data_dir + '/cano_weight_volume.npz')
        pts_lbs = weight_volume.forward_weight(torch.from_numpy(cano_pts)[None].cuda())[0]
    else:
        pts_lbs = interpolate_lbs(cano_pts, cano_smpl_v, smpl_faces, smpl_model.lbs_weights)
        pts_lbs = torch.from_numpy(pts_lbs).cuda()
    np.save(data_dir + '/smpl_pos_map/init_pts_lbs.npy', pts_lbs.cpu().numpy())

    inv_cano_smpl_A = torch.linalg.inv(cano_smpl.A).cuda()
    body_mask = torch.from_numpy(body_mask).cuda()
    cano_pts = torch.from_numpy(cano_pts).cuda()
    pts_lbs = pts_lbs.cuda()

    for pose_idx in tqdm.tqdm(frame_list, desc = 'Generating positional maps...'):
        with torch.no_grad():
            live_smpl_woRoot = smpl_model.forward(
                betas = smpl_data['betas'],
                # global_orient = smpl_data['global_orient'][pose_idx][None],
                # transl = smpl_data['transl'][pose_idx][None],
                body_pose = smpl_data['body_pose'][pose_idx][None],
                jaw_pose = smpl_data['jaw_pose'][pose_idx][None],
                expression = smpl_data['expression'][pose_idx][None],
                # left_hand_pose = smpl_data['left_hand_pose'][pose_idx][None],
                # right_hand_pose = smpl_data['right_hand_pose'][pose_idx][None]
            )

        cano2live_jnt_mats_woRoot = torch.matmul(live_smpl_woRoot.A.cuda(), inv_cano_smpl_A)[0]
        pt_mats = torch.einsum('nj,jxy->nxy', pts_lbs, cano2live_jnt_mats_woRoot)
        live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], cano_pts) + pt_mats[..., :3, 3]
        live_pos_map = torch.zeros((map_size, map_size, 3)).to(live_pts)
        live_pos_map[body_mask] = live_pts
        live_pos_map = F.interpolate(live_pos_map.permute(2, 0, 1)[None], None, [0.5, 0.5], mode = 'nearest')[0]
        live_pos_map = live_pos_map.permute(1, 2, 0).cpu().numpy()

        cv.imwrite(data_dir + '/smpl_pos_map/%08d.exr' % pose_idx, live_pos_map)
