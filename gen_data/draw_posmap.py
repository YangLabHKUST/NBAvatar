import os
import trimesh
import config
import importlib
import yaml
import smplx
import numpy as np
import torch
from network.volume import CanoBlendWeightVolume


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-c', '--config_path', type = str, help = 'Configuration file path.')
    arg_parser.add_argument('--compact', action = 'store_true', default = False, help = 'Use compact pose.')
    arg_parser.add_argument('--compact_scale', type = float, nargs = 3, default = [1.5, 1.15, 1.0], help = 'Compact scale.')
    arg_parser.add_argument('--frame_range', type=int, nargs=3, default = None, help = 'Frame range.')
    args = arg_parser.parse_args()

    opt = yaml.load(open(args.config_path, encoding = 'UTF-8'), Loader = yaml.FullLoader)
    dataset_module = opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
    MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)
    dataset = MvRgbDataset(**opt['train']['data'])
    data_dir, frame_list = dataset.data_dir, dataset.pose_list

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

    template = trimesh.load(data_dir + '/template.ply', process = False)
    using_template = True

    cano_smpl_v = template.vertices.astype(np.float32)
    smpl_faces = template.faces.astype(np.int64)
    if using_template:
        _weight_volume = CanoBlendWeightVolume(data_dir + '/cano_weight_volume.npz')
        _pts_lbs = _weight_volume.forward_weight(torch.from_numpy(cano_smpl_v)[None].cuda())[0]
    else:
        _pts_lbs = smpl_model.lbs_weights.cuda()
    _inv_cano_smpl_A = torch.linalg.inv(cano_smpl.A).cuda()
    with torch.no_grad():
        compact_smpl = smpl_model.forward(
            betas = smpl_data['betas'],
            global_orient = config.cano_smpl_global_orient[None],
            transl = config.cano_smpl_transl[None],
            body_pose = config.compact_smpl_body_pose[None],
        )
        compact_smpl_v = compact_smpl.vertices[0].cpu().numpy()
    cano2compact_jnt_mats = torch.matmul(compact_smpl.A.cuda(), _inv_cano_smpl_A)[0]
    _pt_mats = torch.einsum('nj,jxy->nxy', _pts_lbs, cano2compact_jnt_mats)
    _compact_pts = torch.einsum('nxy,ny->nx', _pt_mats[..., :3, :3], torch.from_numpy(cano_smpl_v).cuda()) + _pt_mats[..., :3, 3]
    _compact_pts = (_compact_pts - torch.from_numpy(cano_center).cuda()) * torch.tensor(args.compact_scale, dtype = torch.float32, device = _compact_pts.device) + torch.from_numpy(cano_center).cuda()
    # render_verts = _compact_pts[smpl_faces.reshape(-1)].cpu().numpy()
    # cano_smpl_v_dup = cano_smpl_v[smpl_faces.reshape(-1)]
    # cano_smpl_n_dup = template.vertex_normals.astype(np.float32)[smpl_faces.reshape(-1)]

    with torch.no_grad():
        live_smpl_woRoot = smpl_model.forward(
            betas = smpl_data['betas'],
            # global_orient = config.cano_smpl_global_orient[None],
            # transl = config.cano_smpl_transl[None],
            body_pose = smpl_data['body_pose'][211][None],
            jaw_pose = smpl_data['jaw_pose'][211][None],
            expression = smpl_data['expression'][211][None],
        )

        cano2live_jnt_mats_woRoot = torch.matmul(live_smpl_woRoot.A.cuda(), _inv_cano_smpl_A)[0]
        pt_mats = torch.einsum('nj,jxy->nxy', _pts_lbs, cano2live_jnt_mats_woRoot)
        live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], torch.from_numpy(cano_smpl_v).cuda()) + pt_mats[..., :3, 3]
        live_pts = live_pts.cpu().numpy()
        colors = (live_pts - live_pts.min(0)) / (live_pts.max(0) - live_pts.min(0))
        _ = trimesh.Trimesh(live_pts, template.faces, vertex_colors=colors, process=False).export('posed_template.ply')

