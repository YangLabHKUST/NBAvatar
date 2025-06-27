from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import *
from easyvolcap.runners.custom_viewer import Viewer # type: ignore
from easyvolcap.models.cameras.optimizable_camera import OptimizableCamera
from easyvolcap.utils.gaussian_utils import convert_to_gaussian_camera
from easyvolcap.utils.data_utils import to_cuda as easy_cuda

import os
from argparse import ArgumentParser
import json 
import torch
import numpy as np
import config
import datetime
import yaml
import importlib
import cv2

from dataset.dataset_pose import PoseDataset
from network.faster_avatar import AvatarNet
from utils import visualize_util
from utils.net_util import to_cuda


def safe_exists(path):
    if path is None:
        return False
    return os.path.exists(path)


def load_ckpt(network, path):
    print('Loading networks from ', path + '/net.pt')
    net_dict = torch.load(path + '/net.pt')
    if 'avatar_net' in net_dict:
        network.load_state_dict(net_dict['avatar_net'])
    else:
        print('[WARNING] Cannot find "avatar_net" from the network checkpoint!')
    epoch_idx = net_dict['epoch_idx']
    iter_idx = net_dict['iter_idx']
    return epoch_idx, iter_idx


# a = '{"H":2032,"W":3840,"K":[[4279.6650390625,0.0,1920.0],[0.0,4279.6650390625,992.4420776367188],[0.0,0.0,1.0]],"R":[[0.41155678033828735,0.911384105682373,0.0],[-0.8666263818740845,0.39134538173675537,0.3095237910747528],[0.2820950746536255,-0.12738661468029022,0.9508903622627258]],"T":[[-4.033830642700195],[-1.7978200912475586],[3.9347341060638428]],"n":0.10000000149011612,"f":1000.0,"t":0.0,"v":0.0,"bounds":[[-10.0,-10.0,-3.0],[10.0,10.0,4.0]],"mass":0.10000000149011612,"moment_of_inertia":0.10000000149011612,"movement_force":1.0,"movement_torque":1.0,"movement_speed":5.0,"origin":[0.0,0.0,0.0],"world_up":[0.0,0.0,-1.0]}'),
class CustomViewer(Viewer):
    def __init__(self,
                 window_size=[1080, 1920],  # height, width
                 window_title: str = f'EasyVolcap Viewer Custom Window',  # MARK: global config
                 fullscreen: bool = False,
                 camera_cfg: dotdict = None,
                 n_frames: int = 1,
                 network: AvatarNet = None,
                 dataset: dotdict = None,
                ):
        super(CustomViewer, self).__init__(
            window_size=window_size,
            window_title=window_title,
            fullscreen=fullscreen,
            camera_cfg=camera_cfg)
        self.network = network
        self.dataset = dataset
        self.n_frames = n_frames

    def custom_render(self, batch):
        index = np.clip(int(batch.t * self.n_frames), 0, self.n_frames - 1)
        data = self.dataset.batches[index]
        batch.update(data)
        gaussian_camera = convert_to_gaussian_camera(
            K=batch.K,
            R=batch.R,
            T=batch.T,
            H=batch.H,
            W=batch.W,
            n=batch.n,
            f=batch.f,
            cpu_K=batch.K,
            cpu_R=batch.R,
            cpu_T=batch.T,
            cpu_H=batch.H,
            cpu_W=batch.W,
            cpu_n=batch.n,
            cpu_f=batch.f,
        )
        batch.update({
            'gaussian_camera': gaussian_camera,
        })
        batch['extr'] = torch.cat([batch.R, batch.T], dim=-1).to('cuda')
        out = self.network.render(batch, bg_color = (1., 1., 1.), use_pca = True, gui = True)
        return out


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--config', type = str, help = 'Configuration file path.')
    args = arg_parser.parse_args()

    config.load_global_opt(args.config)
    config.opt['mode'] = 'test'
    dataset_module = config.opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
    MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)
    training_dataset = MvRgbDataset(**config.opt['train']['data'], training = False)
    if config.opt['test'].get('n_pca', -1) >= 1:
        training_dataset.compute_pca(n_components = config.opt['test']['n_pca'])
    if 'pose_data' in config.opt['test']:
        testing_dataset = PoseDataset(**config.opt['test']['pose_data'], smpl_shape = training_dataset.smpl_data['betas'][0])
        dataset_name = testing_dataset.dataset_name
        seq_name = testing_dataset.seq_name
    else:
        testing_dataset = MvRgbDataset(**config.opt['test']['data'], training = False)
        dataset_name = 'training'
        seq_name = ''
        config.opt['test']['n_pca'] = -1  # cancel PCA for training pose reconstruction

    net = AvatarNet(config.opt['model']).to('cuda')
    net.eval()
    model_path = config.opt['viewer']['exp_dir']
    print(f'Loading checkpoint from {model_path}')
    load_ckpt(net, model_path)
    for p in net.parameters():
        p.requires_grad = False

    use_pca = config.opt['test'].get('n_pca', -1) >= 1
    assert use_pca, 'PCA should be used for viewer'
    item_0 = testing_dataset.getitem(0, training = False)
    object_center = item_0['live_bounds'].mean(0)
    global_orient = item_0['global_orient'].cpu().numpy() if isinstance(item_0['global_orient'], torch.Tensor) else item_0['global_orient']
    global_orient = cv2.Rodrigues(global_orient)[0]
    data_num = len(testing_dataset)

    batches = []
    for idx in tqdm(range(data_num), desc = 'Computing pos maps'):
        extr = visualize_util.calc_free_mv(object_center,
                                            tar_pos = np.array([0, 0, 2.5]),
                                            rot_Y = 0.,
                                            rot_X = 0.,
                                            global_orient = global_orient if config.opt['test'].get('global_orient', False) else None)
        intr = np.array([[1100, 0, 512], [0, 1100, 512], [0, 0, 1]], np.float32)
        img_w = 1024
        img_h = 1024
        getitem_func = testing_dataset.getitem_fast if hasattr(testing_dataset, 'getitem_fast') else testing_dataset.getitem
        with torch.no_grad():
            item = getitem_func(
                idx,
                training = False,
                extr = extr,
                intr = intr,
                img_w = img_w,
                img_h = img_h
            )
        items = to_cuda(item, add_batch = False)

        if 'smpl_pos_map' not in items:
            net.get_pose_map(items)

        if use_pca:
            mask = training_dataset.pos_map_mask
            live_pos_map = items['smpl_pos_map'].permute(1, 2, 0).cpu().numpy()
            front_live_pos_map, back_live_pos_map = np.split(live_pos_map, [3], 2)
            pose_conds = front_live_pos_map[mask]
            new_pose_conds = training_dataset.transform_pca(pose_conds, sigma_pca = float(config.opt['test'].get('sigma_pca', 2.)))
            front_live_pos_map[mask] = new_pose_conds
            live_pos_map = np.concatenate([front_live_pos_map, back_live_pos_map], 2)
            items.update({
                'smpl_pos_map_pca': torch.from_numpy(live_pos_map).to(config.device).permute(2, 0, 1)
            })

        batch = dotdict()
        batch['smpl_pos_map_pca'] = items['smpl_pos_map_pca'][:3]
        batch['cano2live_jnt_mats'] = items['cano2live_jnt_mats']
        # batch['K'] = items['intr']
        # batch['cpu_K'] = items['intr'].cpu()
        # batch['R'] = items['extr'][:3, :3]
        # batch['cpu_R'] = items['extr'][:3, :3].cpu()
        # batch['T'] = items['extr'][:3, 3:]
        # batch['cpu_T'] = items['extr'][:3, 3:].cpu()
        # batch['H'] = torch.tensor(items['img_h'], dtype=torch.int32)
        # batch['W'] = torch.tensor(items['img_w'], dtype=torch.int32)
        # batch['n'] = torch.tensor(0.01, dtype=torch.float32)
        # batch['f'] = torch.tensor(100., dtype=torch.float32)
        # batch['intr'] = items['intr']
        # batch['extr'] = items['extr']
        # batch['img_h'] = items['img_h']
        # batch['img_w'] = items['img_w']
        batches.append(batch)

    del training_dataset
    del testing_dataset

    camera_cfg = dotdict(
        H=items['img_h'],
        W=items['img_w'],
        K=items['intr'].cpu().numpy().tolist(),
        R=items['extr'][:3, :3].cpu().numpy().tolist(),
        T=items['extr'][:3, 3:].cpu().numpy().tolist(),
        n=0.01,
        f=100.,
        t=0.,
        v=0.,
        bounds=[[-10., -10., -10.], [10., 10., 10.]],
        origin=[0., 0., 1.5],
        world_up=[0., 0., 1.],
    )

    dataset = dotdict(
        duration=len(batches) / 30.,
        batches=batches,
    )
    OptimizableCamera.n_frames = len(batches)
    viewer = CustomViewer(
        camera_cfg=camera_cfg,
        network=net,
        dataset=dataset,
        n_frames=len(batches),
    )
    viewer.playing_fps = 30
    viewer.run()
