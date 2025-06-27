import os
import numpy as np
import torch
import yaml
import tqdm
import smplx
import config
import trimesh


if __name__ == '__main__':
    from argparse import ArgumentParser
    import importlib

    arg_parser = ArgumentParser()
    arg_parser.add_argument('-c', '--config_path', type = str, help = 'Configuration file path.')
    args = arg_parser.parse_args()

    opt = yaml.load(open(args.config_path, encoding = 'UTF-8'), Loader = yaml.FullLoader)
    dataset_module = opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
    MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)
    dataset = MvRgbDataset(**opt['train']['data'])
    data_dir, frame_list = dataset.data_dir, dataset.pose_list

    os.makedirs(data_dir + '/smpl_mesh', exist_ok = True)

    smpl_model = smplx.SMPLX(config.PROJ_DIR + '/smpl_files/smplx', gender = 'neutral', use_pca = False, num_pca_comps = 45, flat_hand_mean = True, batch_size = 1)
    smpl_data = np.load(data_dir + '/smpl_params.npz')
    smpl_data = {k: torch.from_numpy(v.astype(np.float32)) for k, v in smpl_data.items()}

    for pose_idx in tqdm.tqdm(frame_list, desc = 'Generating smpl mesh...'):
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
        
        trimesh.Trimesh(
            vertices = live_smpl_woRoot.vertices[0].cpu().numpy(),
            faces = smpl_model.faces,
            process = False
        ).export(data_dir + '/smpl_mesh/%08d.ply' % pose_idx)