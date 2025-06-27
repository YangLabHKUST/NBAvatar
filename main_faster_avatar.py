import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import yaml
import shutil
import collections
import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import cv2 as cv
import glob
import datetime
import trimesh
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import importlib
import config
from network.faster_avatar import AvatarNet
from network.lpips import LPIPS
from dataset.dataset_pose import PoseDataset
import utils.net_util as net_util
import utils.visualize_util as visualize_util
from utils.renderer import Renderer
from utils.net_util import to_cuda
from utils.obj_io import save_mesh_as_ply
from utils.obj_io import export_pts
from utils.sh_utils import SH2RGB


def gaussian_scaling_loss(scaling, threshold=0.01):
    scale_sub = scaling - threshold
    loss = torch.where(scale_sub > 0, scaling, torch.tensor(0, device=scaling.device)).mean()
    return loss


def safe_exists(path):
    if path is None:
        return False
    return os.path.exists(path)


class AvatarTrainer:
    def __init__(self, opt):
        self.opt = opt
        self.patch_size = 512
        self.iter_idx = 0
        self.iter_num = self.opt['train'].get('iter_num', 800000)
        self.lr_init = float(self.opt['train'].get('lr_init', 5e-4))
        self.iteration_lpips_random_patch = self.opt['train'].get('iteration_lpips_random_patch', 200000)
        self.iteration_lpips_start = self.opt['train'].get('iteration_lpips_start', 3000)
        self.gs_lr_scale = self.opt['train'].get('gs_lr_scale', 1.0)
        self.pretrain_iter_num = self.opt['train'].get('pretrain_iter_num', 2000)
        self.pretrain_color = self.opt['train'].get('pretrain_color', True)

        self.avatar_net = AvatarNet(self.opt['model']).to(config.device)
        print(self.avatar_net)
        self.lr_dict = {
            'net': self.lr_init,
            'basis': self.lr_init * 0.2,
        }
        if self.avatar_net.opt_cano_gs:
            self.lr_dict.update({
                'xyz': self.opt['train'].get('lr_xyz', 0.000016) * self.gs_lr_scale,
                'offset': self.opt['train'].get('lr_offset', 0.001) * self.gs_lr_scale,
                'opacity': self.opt['train'].get('lr_opacity', 0.005) * self.gs_lr_scale,
                'scale': self.opt['train'].get('lr_scale', 0.0005) * self.gs_lr_scale,
                'rotation': self.opt['train'].get('lr_rotation', 0.0001) * self.gs_lr_scale,
                'features_dc': self.opt['train'].get('lr_features_dc', 0.00025) * self.gs_lr_scale,
                'features_rest': self.opt['train'].get('lr_features_rest', 0.00025) * self.gs_lr_scale / 20,
            })
        print(self.lr_dict)

        param_groups = [
            {'name': 'net', 'params': [p for n, p in self.avatar_net.named_parameters() if 'net' in n and p.requires_grad], 'lr': self.lr_dict['net']},
            {'name': 'basis', 'params': [p for n, p in self.avatar_net.named_parameters() if 'basis' in n and p.requires_grad], 'lr': self.lr_dict['basis']},
        ]
        if self.avatar_net.opt_cano_gs:
            param_groups.extend([
                {'name': 'xyz', 'params': [p for n, p in self.avatar_net.named_parameters() if '_xyz' in n and p.requires_grad], 'lr': self.lr_dict['xyz']},
                {'name': 'offset', 'params': [p for n, p in self.avatar_net.named_parameters() if '_offset' in n and p.requires_grad], 'lr': self.lr_dict['offset']},
                {'name': 'opacity', 'params': [p for n, p in self.avatar_net.named_parameters() if '_opacity' in n and p.requires_grad], 'lr': self.lr_dict['opacity']},
                {'name': 'scale', 'params': [p for n, p in self.avatar_net.named_parameters() if '_scaling' in n and p.requires_grad], 'lr': self.lr_dict['scale']},
                {'name': 'rotation', 'params': [p for n, p in self.avatar_net.named_parameters() if '_rotation' in n and p.requires_grad], 'lr': self.lr_dict['rotation']},
                {'name': 'features_dc', 'params': [p for n, p in self.avatar_net.named_parameters() if '_features_dc' in n and p.requires_grad], 'lr': self.lr_dict['features_dc']},
                {'name': 'features_rest', 'params': [p for n, p in self.avatar_net.named_parameters() if '_features_rest' in n and p.requires_grad], 'lr': self.lr_dict['features_rest']},
            ])
        for param_group in param_groups:
            print('param_group:', param_group['name'], len(param_group['params']))
        self.optm = torch.optim.Adam(
            param_groups, lr = self.lr_init
        )

        self.random_bg_color = self.opt['train'].get('random_bg_color', True)
        self.bg_color = (1., 1., 1.)
        self.bg_color_cuda = torch.from_numpy(np.asarray(self.bg_color)).to(torch.float32).to(config.device)
        self.loss_weight = self.opt['train']['loss_weight']
        self.finetune_color = self.opt['train']['finetune_color']

        print('# Parameter number of AvatarNet is %d' % (sum([p.numel() for p in self.avatar_net.parameters() if p.requires_grad])))

    def update_lr(self):
        alpha = 0.05
        progress = self.iter_idx / self.iter_num
        learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        for param_group in self.optm.param_groups:
            lr_init = self.lr_dict[param_group['name']]
            param_group['lr'] = lr_init * learning_factor
        return self.lr_init * learning_factor

    @staticmethod
    def requires_net_grad(net: torch.nn.Module, flag = True):
        for p in net.parameters():
            p.requires_grad = flag

    def crop_image(self, gt_mask, patch_size, randomly, bg_color, *args):
        """
        :param gt_mask: (H, W)
        :param patch_size: resize the cropped patch to the given patch_size
        :param randomly: whether to randomly sample the patch
        :param args: input images with shape of (C, H, W)
        """
        mask_uv = torch.argwhere(gt_mask > 0.)
        min_v, min_u = mask_uv.min(0)[0]
        max_v, max_u = mask_uv.max(0)[0]
        len_v = max_v - min_v
        len_u = max_u - min_u
        max_size = max(len_v, len_u)

        cropped_images = []
        if randomly and max_size > patch_size:
            random_v = torch.randint(0, max_size - patch_size + 1, (1,)).to(max_size)
            random_u = torch.randint(0, max_size - patch_size + 1, (1,)).to(max_size)
        for image in args:
            cropped_image = bg_color[:, None, None] * torch.ones((3, max_size, max_size), dtype = image.dtype, device = image.device)
            if len_v > len_u:
                start_u = (max_size - len_u) // 2
                cropped_image[:, :, start_u: start_u + len_u] = image[:, min_v: max_v, min_u: max_u]
            else:
                start_v = (max_size - len_v) // 2
                cropped_image[:, start_v: start_v + len_v, :] = image[:, min_v: max_v, min_u: max_u]

            if randomly and max_size > patch_size:
                cropped_image = cropped_image[:, random_v: random_v + patch_size, random_u: random_u + patch_size]
            else:
                cropped_image = F.interpolate(cropped_image[None], size = (patch_size, patch_size), mode = 'bilinear')[0]
            cropped_images.append(cropped_image)

        # cv.imshow('cropped_image', cropped_image.detach().cpu().numpy().transpose(1, 2, 0))
        # cv.imshow('cropped_gt_image', cropped_gt_image.detach().cpu().numpy().transpose(1, 2, 0))
        # cv.waitKey(0)

        if len(cropped_images) > 1:
            return cropped_images
        else:
            return cropped_images[0]

    def compute_lpips_loss(self, image, gt_image):
        assert image.shape[1] == image.shape[2] and gt_image.shape[1] == gt_image.shape[2]
        lpips_loss = self.lpips.forward(
            image[None, [2, 1, 0]],
            gt_image[None, [2, 1, 0]],
            normalize = True
        ).mean()
        return lpips_loss

    def forward_one_pass_pretrain(self, items):
        total_loss = 0
        batch_losses = {}
        l1_loss = torch.nn.L1Loss()

        items = net_util.delete_batch_idx(items)
        pose_map = items['smpl_pos_map'][:3]

        if self.avatar_net.share_blend_weight:
            blend_weight, _ = self.avatar_net.color_bw_net([self.avatar_net.color_style], pose_map[None], randomize_noise = False)
        else:
            blend_weight = None
        positions, _ = self.avatar_net.get_geometry(pose_map, return_delta = True, blend_weight = blend_weight)
        opacities, scales, rotations, colors, _ = self.avatar_net.get_colors(pose_map, return_delta = True, blend_weight = blend_weight)

        position_loss = l1_loss(positions, torch.zeros_like(positions))
        total_loss += position_loss
        batch_losses.update({
            'position': position_loss.item()
        })

        opacity_loss = l1_loss(opacities, torch.zeros_like(opacities))
        total_loss += opacity_loss
        batch_losses.update({
            'opacity': opacity_loss.item()
        })

        scale_loss = l1_loss(scales, torch.zeros_like(scales))
        total_loss += scale_loss
        batch_losses.update({
            'scale': scale_loss.item()
        })

        rotation_loss = l1_loss(rotations, torch.zeros_like(rotations))
        total_loss += rotation_loss
        batch_losses.update({
            'rotation': rotation_loss.item()
        })

        if self.avatar_net.pred_color_offset:
            color_loss = l1_loss(colors, torch.zeros_like(colors))
            total_loss += color_loss
            batch_losses.update({
                'color': color_loss.item()
            })
            gaussian_scales = self.avatar_net.cano_gaussian_model.get_scaling
            scale_loss = l1_loss(gaussian_scales, torch.zeros_like(gaussian_scales))
            total_loss += scale_loss
            batch_losses.update({
                'scale_loss': scale_loss.item()
            })
        
        if self.pretrain_color:
            if self.random_bg_color:
                bg_color_cuda = torch.rand(3, dtype = torch.float32, device = config.device)
            else:
                bg_color_cuda = torch.from_numpy(np.asarray(self.bg_color)).to(torch.float32).to(config.device)

            render_output = self.avatar_net.render(items, bg_color_cuda, sh_degree=0)
            image = render_output['rgb_map'].permute(2, 0, 1)

            # mask image & set bg color
            items['color_img'][~items['mask_img']] = bg_color_cuda
            gt_image = items['color_img'].permute(2, 0, 1)
            # mask_img = items['mask_img'].to(torch.float32)
            boundary_mask_img = 1. - items['boundary_mask_img'].to(torch.float32)
            image = image * boundary_mask_img[None] + (1. - boundary_mask_img[None]) * bg_color_cuda[:, None, None]
            gt_image = gt_image * boundary_mask_img[None] + (1. - boundary_mask_img[None]) * bg_color_cuda[:, None, None]
            # gt_image = items['color_img']
            # mask_img = items['mask_img'].to(torch.bool)
            # mask_boundary_img = items['boundary_mask_img']
            # gt_image[~mask_img] = bg_color_cuda
            # gt_image[mask_boundary_img] = bg_color_cuda
            # image[mask_boundary_img] = bg_color_cuda

            # debug
            # image_np = image.detach().cpu().numpy()
            # gt_image_np = gt_image.detach().cpu().numpy()
            # cv.imwrite(f'debug/image_{self.iter_idx}.png', (image_np * 255).astype(np.uint8))
            # cv.imwrite(f'debug/gt_image_{self.iter_idx}.png', (gt_image_np * 255).astype(np.uint8))

            l1_loss = torch.abs(image - gt_image).mean()
            total_loss += l1_loss
            batch_losses.update({
                'l1_loss': l1_loss.item()
            })

            # rendered_mask = render_output['mask_map'].squeeze(-1) * boundary_mask_img
            # gt_mask = mask_img * boundary_mask_img
            # # cv.imwrite(f'rendered_mask_{self.iter_idx}.png', (rendered_mask.detach().cpu().numpy() * 255).astype(np.uint8))
            # # cv.imwrite(f'gt_mask_{self.iter_idx}.png', (gt_mask.detach().cpu().numpy() * 255).astype(np.uint8))
            # # cv.waitKey(0)
            # mask_loss = torch.abs(rendered_mask - gt_mask).mean()
            # # mask_loss = torch.nn.BCELoss()(rendered_mask, gt_mask)
            # total_loss += 0.01 * mask_loss
            # batch_losses.update({
            #     'mask_loss': mask_loss.item()
            # })

        total_loss.backward()

        self.optm.step()
        self.optm.zero_grad()

        return total_loss, batch_losses

    def forward_one_pass(self, items):
        # forward_start = torch.cuda.Event(enable_timing = True)
        # forward_end = torch.cuda.Event(enable_timing = True)
        # backward_start = torch.cuda.Event(enable_timing = True)
        # backward_end = torch.cuda.Event(enable_timing = True)
        # step_start = torch.cuda.Event(enable_timing = True)
        # step_end = torch.cuda.Event(enable_timing = True)

        if self.random_bg_color:
            bg_color_cuda = torch.rand(3, dtype = torch.float32, device = config.device)
        else:
            bg_color_cuda = torch.from_numpy(np.asarray(self.bg_color)).to(torch.float32).to(config.device)

        total_loss = 0
        batch_losses = {}

        items = net_util.delete_batch_idx(items)

        """ Optimize generator """
        if self.finetune_color:
            raise NotImplementedError('Finetune color is not implemented!')
            self.requires_net_grad(self.avatar_net.color_net, True)
            self.requires_net_grad(self.avatar_net.position_net, False)
            self.requires_net_grad(self.avatar_net.other_net, True)
        else:
            pass
            # self.requires_net_grad(self.avatar_net, True)

        # forward_start.record()
        if self.avatar_net.max_sh_degree > 0:
            current_sh_degree = min(self.iter_idx // 100000, self.avatar_net.max_sh_degree)
        else:
            current_sh_degree = None
        render_output = self.avatar_net.render(items, bg_color_cuda, sh_degree=current_sh_degree)
        image = render_output['rgb_map'].permute(2, 0, 1)

        # mask image & set bg color
        items['color_img'][~items['mask_img']] = bg_color_cuda
        gt_image = items['color_img'].permute(2, 0, 1)
        mask_img = items['mask_img'].to(torch.float32)
        boundary_mask_img = 1. - items['boundary_mask_img'].to(torch.float32)
        image = image * boundary_mask_img[None] + (1. - boundary_mask_img[None]) * bg_color_cuda[:, None, None]
        gt_image = gt_image * boundary_mask_img[None] + (1. - boundary_mask_img[None]) * bg_color_cuda[:, None, None]
        # gt_image = items['color_img']
        # mask_img = items['mask_img'].to(torch.bool)
        # mask_boundary_img = items['boundary_mask_img']
        # gt_image[~mask_img] = bg_color_cuda
        # gt_image[mask_boundary_img] = bg_color_cuda
        # image[mask_boundary_img] = bg_color_cuda
        # image = image.permute(2, 0, 1)
        # gt_image = gt_image.permute(2, 0, 1)

        # debug
        # cv.imshow('image', image.detach().permute(1, 2, 0).cpu().numpy())
        # cv.imshow('gt_image', gt_image.permute(1, 2, 0).cpu().numpy())
        # cv.waitKey(0)

        if self.loss_weight['l1'] > 0.:
            l1_loss = torch.abs(image - gt_image).mean()
            total_loss += self.loss_weight['l1'] * l1_loss
            batch_losses.update({
                'l1_loss': l1_loss.item()
            })

        if self.loss_weight.get('mask', 0.) and 'mask_map' in render_output:
            raise NotImplementedError('Mask loss is not implemented!')
            rendered_mask = render_output['mask_map'].squeeze(-1) * boundary_mask_img
            gt_mask = mask_img * boundary_mask_img
            # cv.imshow('rendered_mask', rendered_mask.detach().cpu().numpy())
            # cv.imshow('gt_mask', gt_mask.detach().cpu().numpy())
            # cv.waitKey(0)
            mask_loss = torch.abs(rendered_mask - gt_mask).mean()
            # mask_loss = torch.nn.BCELoss()(rendered_mask, gt_mask)
            total_loss += self.loss_weight.get('mask', 0.) * mask_loss
            batch_losses.update({
                'mask_loss': mask_loss.item()
            })

        if self.loss_weight['lpips'] > 0.:
            # crop images
            random_patch_flag = False if self.iter_idx < self.iteration_lpips_random_patch else True
            image, gt_image = self.crop_image(mask_img, 512, random_patch_flag, bg_color_cuda, image, gt_image)
            # cv.imshow('image', image.detach().permute(1, 2, 0).cpu().numpy())
            # cv.imshow('gt_image', gt_image.permute(1, 2, 0).cpu().numpy())
            # cv.waitKey(0)
            if self.iter_idx > self.iteration_lpips_start:
                lpips_loss = self.compute_lpips_loss(image, gt_image)
            else:
                lpips_loss = torch.tensor(0.)
            total_loss += self.loss_weight['lpips'] * lpips_loss
            batch_losses.update({
                'lpips_loss': lpips_loss.item()
            })

        if self.loss_weight['scaling'] > 0.:
            scaling_loss = gaussian_scaling_loss(render_output['posed_gaussians']['scales'], 0.01)
            total_loss += self.loss_weight['scaling'] * scaling_loss
            batch_losses.update({
                'scaling_loss': scaling_loss.item()
            })

        if self.loss_weight['offset'] > 0. and 'offset' in render_output:
            offset = render_output['offset']
            offset_loss = torch.linalg.norm(offset, dim = -1).mean()
            total_loss += self.loss_weight['offset'] * offset_loss
            batch_losses.update({
                'offset_loss': offset_loss.item()
            })
        
        if self.loss_weight['cano_offset'] > 0. and 'cano_offset' in render_output:
            cano_offset = render_output['cano_offset']
            cano_offset_loss = torch.linalg.norm(cano_offset, dim = -1).mean()
            total_loss += self.loss_weight['cano_offset'] * cano_offset_loss
            batch_losses.update({
                'cano_offset_loss': cano_offset_loss.item()
            })

        if self.loss_weight.get('pos_tv', 0.) > 0. and 'geo_bw_feat' in render_output:
            geo_bw_feat = render_output['geo_bw_feat']
            tv_h = torch.pow(geo_bw_feat[:, :, 1:, :] - geo_bw_feat[:, :, :-1, :], 2).sum()
            tv_w = torch.pow(geo_bw_feat[:, :, :, 1:] - geo_bw_feat[:, :, :, :-1], 2).sum()
            pos_tv = (tv_h + tv_w) / np.prod(geo_bw_feat.shape)
            total_loss += self.loss_weight['pos_tv'] * pos_tv
            batch_losses.update({
                'pos_tv': pos_tv.item()
            })
        
        if self.loss_weight.get('rgb_tv', 0.) > 0. and 'rgb_bw_feat' in render_output:
            rgb_bw_feat = render_output['rgb_bw_feat']
            tv_h = torch.pow(rgb_bw_feat[:, :, 1:, :] - rgb_bw_feat[:, :, :-1, :], 2).sum()
            tv_w = torch.pow(rgb_bw_feat[:, :, :, 1:] - rgb_bw_feat[:, :, :, :-1], 2).sum()
            rgb_tv = (tv_h + tv_w) / np.prod(rgb_bw_feat.shape)
            total_loss += self.loss_weight['rgb_tv'] * rgb_tv
            batch_losses.update({
                'rgb_tv': rgb_tv.item()
            })
        
        # forward_end.record()

        # backward_start.record()
        total_loss.backward()
        # backward_end.record()

        # step_start.record()
        self.optm.step()
        self.optm.zero_grad()
        # step_end.record()

        # torch.cuda.synchronize()
        # print(f'Forward costs: {forward_start.elapsed_time(forward_end) / 1000.}, ',
        #       f'Backward costs: {backward_start.elapsed_time(backward_end) / 1000.}, ',
        #       f'Step costs: {step_start.elapsed_time(step_end) / 1000.}')

        return total_loss, batch_losses

    def pretrain(self):
        dataset_module = self.opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
        MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)
        self.dataset = MvRgbDataset(**self.opt['train']['data'])
        batch_size = self.opt['train']['batch_size']
        num_workers = self.opt['train']['num_workers']
        batch_num = len(self.dataset) // batch_size
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size = batch_size,
                                                 shuffle = True,
                                                 num_workers = num_workers,
                                                 drop_last = True)

        # tb writer
        log_dir = self.opt['train']['net_ckpt_dir'] + '/' + 'pretrain'
        writer = SummaryWriter(log_dir)
        smooth_interval = 10
        smooth_count = 0
        smooth_losses = {}

        for epoch_idx in range(0, 9999999):
            self.epoch_idx = epoch_idx
            for batch_idx, items in enumerate(dataloader):
                self.iter_idx = batch_idx + epoch_idx * batch_num
                items = to_cuda(items)

                # one_step_start.record()
                total_loss, batch_losses = self.forward_one_pass_pretrain(items)
                # one_step_end.record()
                # torch.cuda.synchronize()
                # print('One step costs %f secs' % (one_step_start.elapsed_time(one_step_end) / 1000.))

                # record batch loss
                for key, loss in batch_losses.items():
                    if key in smooth_losses:
                        smooth_losses[key] += loss
                    else:
                        smooth_losses[key] = loss
                smooth_count += 1

                if self.iter_idx % smooth_interval == 0:
                    log_info = 'epoch %d, batch %d, iter %d, ' % (epoch_idx, batch_idx, self.iter_idx)
                    for key in smooth_losses.keys():
                        smooth_losses[key] /= smooth_count
                        writer.add_scalar('%s/Iter' % key, smooth_losses[key], self.iter_idx)
                        log_info = log_info + ('%s: %f, ' % (key, smooth_losses[key]))
                        smooth_losses[key] = 0.
                    smooth_count = 0
                    print(log_info)
                    with open(os.path.join(log_dir, 'loss.txt'), 'a') as fp:
                        fp.write(log_info + '\n')

                if self.iter_idx % 200 == 0 and self.iter_idx != 0:
                    self.mini_test(pretraining = True)

                if self.iter_idx == self.pretrain_iter_num:
                    model_folder = self.opt['train']['net_ckpt_dir'] + '/pretrained'
                    os.makedirs(model_folder, exist_ok = True)
                    self.save_ckpt(model_folder, save_optm = True)
                    self.iter_idx = 0
                    return

    def train(self):
        dataset_module = self.opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
        MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)
        self.dataset = MvRgbDataset(**self.opt['train']['data'])
        batch_size = self.opt['train']['batch_size']
        num_workers = self.opt['train']['num_workers']
        batch_num = len(self.dataset) // batch_size
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size = batch_size,
                                                 shuffle = True,
                                                 num_workers = num_workers,
                                                 drop_last = True)

        if 'lpips' in self.opt['train']['loss_weight']:
            self.lpips = LPIPS(net = 'vgg').to(config.device)
            for p in self.lpips.parameters():
                p.requires_grad = False

        if self.opt['train']['prev_ckpt'] and os.path.exists(self.opt['train']['prev_ckpt']):
            print(f'Loading checkpoint from {self.opt["train"]["prev_ckpt"]}')
            start_epoch, self.iter_idx = self.load_ckpt(self.opt['train']['prev_ckpt'], load_optm = True)
            start_epoch += 1
            self.iter_idx += 1
        else:
            prev_ckpt_path = self.opt['train']['net_ckpt_dir'] + '/epoch_latest'
            if safe_exists(prev_ckpt_path):
                print(f'Loading checkpoint from {prev_ckpt_path}')
                start_epoch, self.iter_idx = self.load_ckpt(prev_ckpt_path, load_optm = True)
                start_epoch += 1
                self.iter_idx += 1
            else:
                if safe_exists(self.opt['train']['pretrained_dir']):
                    print(f'Loading pretrained checkpoint from {self.opt["train"]["pretrained_dir"]}')
                    self.load_ckpt(self.opt['train']['pretrained_dir'], load_optm = False)
                elif safe_exists(self.opt['train']['net_ckpt_dir'] + '/pretrained'):
                    print(f'Loading pretrained checkpoint from {self.opt["train"]["net_ckpt_dir"] + "/pretrained"}')
                    self.load_ckpt(self.opt['train']['net_ckpt_dir'] + '/pretrained', load_optm = False)
                else:
                    raise FileNotFoundError('Cannot find pretrained checkpoint!')

                self.optm.state = collections.defaultdict(dict)
                start_epoch = 0
                self.iter_idx = 0

        # one_step_start = torch.cuda.Event(enable_timing = True)
        # one_step_end = torch.cuda.Event(enable_timing = True)

        # tb writer
        log_dir = self.opt['train']['net_ckpt_dir'] + '/' + 'train'
        writer = SummaryWriter(log_dir)
        yaml.dump(self.opt, open(log_dir + '/config_bk.yaml', 'w'), sort_keys = False)
        smooth_interval = 10
        smooth_count = 0
        smooth_losses = {}

        for epoch_idx in range(start_epoch, 9999999):
            self.epoch_idx = epoch_idx
            for batch_idx, items in enumerate(dataloader):
                lr = self.update_lr()

                items = to_cuda(items)

                # one_step_start.record()
                total_loss, batch_losses = self.forward_one_pass(items)
                # one_step_end.record()
                # torch.cuda.synchronize()
                # print('One step costs %f secs' % (one_step_start.elapsed_time(one_step_end) / 1000.))

                # record batch loss
                for key, loss in batch_losses.items():
                    if key in smooth_losses:
                        smooth_losses[key] += loss
                    else:
                        smooth_losses[key] = loss
                smooth_count += 1

                if self.iter_idx % smooth_interval == 0:
                    log_info = 'epoch %d, batch %d, iter %d, lr %e, ' % (epoch_idx, batch_idx, self.iter_idx, lr)
                    for key in smooth_losses.keys():
                        smooth_losses[key] /= smooth_count
                        writer.add_scalar('%s/Iter' % key, smooth_losses[key], self.iter_idx)
                        log_info = log_info + ('%s: %f, ' % (key, smooth_losses[key]))
                        smooth_losses[key] = 0.
                    smooth_count = 0
                    print(log_info)
                    with open(os.path.join(log_dir, 'loss.txt'), 'a') as fp:
                        fp.write(log_info + '\n')
                    torch.cuda.empty_cache()

                if self.iter_idx % self.opt['train']['eval_interval'] == 0 and self.iter_idx != 0:
                    if self.iter_idx % (10 * self.opt['train']['eval_interval']) == 0:
                        eval_cano_pts = True
                    else:
                        eval_cano_pts = False
                    self.mini_test(eval_cano_pts = eval_cano_pts)

                if self.iter_idx % self.opt['train']['ckpt_interval']['batch'] == 0 and self.iter_idx != 0:
                    for folder in glob.glob(self.opt['train']['net_ckpt_dir'] + '/batch_*'):
                        shutil.rmtree(folder)
                    model_folder = self.opt['train']['net_ckpt_dir'] + '/batch_%d' % self.iter_idx
                    os.makedirs(model_folder, exist_ok = True)
                    self.save_ckpt(model_folder, save_optm = True)

                if self.iter_idx == self.iter_num:
                    print('# Training is done.')
                    return

                self.iter_idx += 1

            """ End of epoch """
            if epoch_idx % self.opt['train']['ckpt_interval']['epoch'] == 0 and epoch_idx != 0:
                model_folder = self.opt['train']['net_ckpt_dir'] + '/epoch_%d' % epoch_idx
                os.makedirs(model_folder, exist_ok = True)
                self.save_ckpt(model_folder)

            if batch_num > 50:
                latest_folder = self.opt['train']['net_ckpt_dir'] + '/epoch_latest'
                os.makedirs(latest_folder, exist_ok = True)
                self.save_ckpt(latest_folder)

    @torch.no_grad()
    def mini_test(self, pretraining = False, eval_cano_pts = False):
        self.avatar_net.eval()

        img_factor = self.opt['train'].get('eval_img_factor', 1.0)
        # training data
        pose_idx, view_idx = self.opt['train'].get('eval_training_ids', (310, 19))
        intr = self.dataset.intr_mats[view_idx].copy()
        intr[:2] *= img_factor
        item = self.dataset.getitem(0,
                                    pose_idx = pose_idx,
                                    view_idx = view_idx,
                                    training = False,
                                    eval = True,
                                    img_h = int(self.dataset.img_heights[view_idx] * img_factor),
                                    img_w = int(self.dataset.img_widths[view_idx] * img_factor),
                                    extr = self.dataset.extr_mats[view_idx],
                                    intr = intr,
                                    exact_hand_pose = True)
        items = net_util.to_cuda(item, add_batch = False)

        if self.avatar_net.max_sh_degree > 0:
            current_sh_degree = min(self.iter_idx // 100000, self.avatar_net.max_sh_degree)
        else:
            current_sh_degree = None
        gs_render = self.avatar_net.render(items, bg_color = self.bg_color_cuda, sh_degree=current_sh_degree)
        # gs_render = self.avatar_net.render_debug(items)
        rgb_map = gs_render['rgb_map']
        rgb_map.clip_(0., 1.)
        rgb_map = (rgb_map.cpu().numpy() * 255).astype(np.uint8)
        # cv.imshow('rgb_map', rgb_map.cpu().numpy())
        # cv.waitKey(0)
        if not pretraining:
            output_dir = self.opt['train']['net_ckpt_dir'] + '/eval/training'
        else:
            output_dir = self.opt['train']['net_ckpt_dir'] + '/eval_pretrain/training'
        gt_image, _ = self.dataset.load_color_mask_images(pose_idx, view_idx)
        if gt_image is not None:
            gt_image = cv.resize(gt_image, (0, 0), fx = img_factor, fy = img_factor)
            rgb_map = np.concatenate([rgb_map, gt_image], 1)
        os.makedirs(output_dir, exist_ok = True)
        cv.imwrite(output_dir + '/iter_%d.jpg' % self.iter_idx, rgb_map)
        if eval_cano_pts:
            os.makedirs(output_dir + '/cano_pts', exist_ok = True)
            save_mesh_as_ply(output_dir + '/cano_pts/iter_%d.ply' % self.iter_idx, (self.avatar_net.cano_gaussian_model.get_xyz + gs_render['offset']).cpu().numpy())

        # training data
        pose_idx, view_idx = self.opt['train'].get('eval_testing_ids', (310, 19))
        intr = self.dataset.intr_mats[view_idx].copy()
        intr[:2] *= img_factor
        item = self.dataset.getitem(0,
                                    pose_idx = pose_idx,
                                    view_idx = view_idx,
                                    training = False,
                                    eval = True,
                                    img_h = int(self.dataset.img_heights[view_idx] * img_factor),
                                    img_w = int(self.dataset.img_widths[view_idx] * img_factor),
                                    extr = self.dataset.extr_mats[view_idx],
                                    intr = intr,
                                    exact_hand_pose = True)
        items = net_util.to_cuda(item, add_batch = False)

        if self.avatar_net.max_sh_degree > 0:
            current_sh_degree = min(self.iter_idx // 100000, self.avatar_net.max_sh_degree)
        else:
            current_sh_degree = None
        gs_render = self.avatar_net.render(items, bg_color = self.bg_color_cuda, sh_degree=current_sh_degree)
        # gs_render = self.avatar_net.render_debug(items)
        rgb_map = gs_render['rgb_map']
        rgb_map.clip_(0., 1.)
        rgb_map = (rgb_map.cpu().numpy() * 255).astype(np.uint8)
        # cv.imshow('rgb_map', rgb_map.cpu().numpy())
        # cv.waitKey(0)
        if not pretraining:
            output_dir = self.opt['train']['net_ckpt_dir'] + '/eval/testing'
        else:
            output_dir = self.opt['train']['net_ckpt_dir'] + '/eval_pretrain/testing'
        gt_image, _ = self.dataset.load_color_mask_images(pose_idx, view_idx)
        if gt_image is not None:
            gt_image = cv.resize(gt_image, (0, 0), fx = img_factor, fy = img_factor)
            rgb_map = np.concatenate([rgb_map, gt_image], 1)
        os.makedirs(output_dir, exist_ok = True)
        cv.imwrite(output_dir + '/iter_%d.jpg' % self.iter_idx, rgb_map)
        if eval_cano_pts:
            os.makedirs(output_dir + '/cano_pts', exist_ok = True)
            save_mesh_as_ply(output_dir + '/cano_pts/iter_%d.ply' % self.iter_idx, (self.avatar_net.cano_gaussian_model.get_xyz + gs_render['offset']).cpu().numpy())

        self.avatar_net.train()

    @torch.no_grad()
    def test(self):
        self.avatar_net.eval()

        dataset_module = self.opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
        MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)
        training_dataset = MvRgbDataset(**self.opt['train']['data'], training = False)
        if self.opt['test'].get('n_pca', -1) >= 1:
            training_dataset.compute_pca(n_components = self.opt['test']['n_pca'])
        if 'pose_data' in self.opt['test']:
            testing_dataset = PoseDataset(**self.opt['test']['pose_data'], smpl_shape = training_dataset.smpl_data['betas'][0])
            dataset_name = testing_dataset.dataset_name
            seq_name = testing_dataset.seq_name
        else:
            testing_dataset = MvRgbDataset(**self.opt['test']['data'], training = False)
            dataset_name = 'training'
            seq_name = ''

            self.opt['test']['n_pca'] = -1  # cancel PCA for training pose reconstruction

        self.dataset = testing_dataset
        if os.path.exists(self.opt['test']['prev_ckpt']):
            print(f'Loading checkpoint from {self.opt["test"]["prev_ckpt"]}')
            iter_idx = self.load_ckpt(self.opt['test']['prev_ckpt'], False)[1]
        elif os.path.exists(os.path.join(self.opt['train']['net_ckpt_dir'], 'net.pt')):
            print(f'Loading checkpoint from {self.opt["train"]["net_ckpt_dir"]}/net.pt')
            iter_idx = self.load_ckpt(os.path.join(self.opt['train']['net_ckpt_dir']), False)[1]
        else:
            prev_ckpt_path = self.opt['train']['net_ckpt_dir'] + '/epoch_latest'
            if safe_exists(prev_ckpt_path):
                print(f'Loading checkpoint from {prev_ckpt_path}')
                iter_idx = self.load_ckpt(prev_ckpt_path, False)[1]
            else:
                print(f'Cannot find checkpoint from {self.opt["test"]["prev_ckpt"]}')
                raise FileNotFoundError('Cannot find checkpoint!')

        output_dir = self.opt['test'].get('output_dir', None)
        if output_dir is None:
            view_setting = config.opt['test'].get('view_setting', 'free')
            if view_setting == 'camera':
                view_folder = 'cam_%03d' % config.opt['test']['render_view_idx']
            else:
                view_folder = view_setting + '_view'
            exp_name = os.path.basename(os.path.dirname(self.opt['test']['prev_ckpt']))
            output_dir = f'{self.opt["train"]["net_ckpt_dir"]}/{training_dataset.subject_name}/{exp_name}/{dataset_name}_{seq_name}_{view_folder}' + '/batch_%06d' % iter_idx

        use_pca = self.opt['test'].get('n_pca', -1) >= 1
        if use_pca:
            output_dir += '/pca_%d_sigma_%.2f' % (self.opt['test'].get('n_pca', -1), float(self.opt['test'].get('sigma_pca', 1.)))
        else:
            output_dir += '/vanilla'
        print('# Output dir: \033[1;31m%s\033[0m' % output_dir)

        # os.makedirs(output_dir + '/live_skeleton', exist_ok = True)
        # os.makedirs(output_dir + '/rgb_map', exist_ok = True)
        # os.makedirs(output_dir + '/mask_map', exist_ok = True)

        geo_renderer = None
        item_0 = self.dataset.getitem(0, training = False)
        object_center = item_0['live_bounds'].mean(0)
        global_orient = item_0['global_orient'].cpu().numpy() if isinstance(item_0['global_orient'], torch.Tensor) else item_0['global_orient']
        global_orient = cv.Rodrigues(global_orient)[0]
        # print('object_center: ', object_center.tolist())
        # print('global_orient: ', global_orient.tolist())
        # # exit(1)

        time_start = torch.cuda.Event(enable_timing = True)
        time_start_all = torch.cuda.Event(enable_timing = True)
        time_end = torch.cuda.Event(enable_timing = True)

        data_num = len(self.dataset)
        if self.opt['test'].get('fix_hand', False):
            self.avatar_net.generate_mean_hands()
        log_time = True

        # if self.opt['test'].get('save_cameras', False):
        save_cameras = True
        if save_cameras:
            test_cameras = dict(
                intr=[],
                extr=[],
                img_w=[],
                img_h=[],
                pose_idx=[],
            )
        else:
            test_cameras = None

        for idx in tqdm(range(data_num), desc = 'Rendering avatars...'):
            if log_time:
                time_start.record()
                time_start_all.record()

            img_scale = self.opt['test'].get('img_scale', 1.0)
            view_setting = config.opt['test'].get('view_setting', 'free')
            if view_setting == 'camera':
                # training view setting
                cam_id = config.opt['test']['render_view_idx']
                intr = self.dataset.intr_mats[cam_id].copy()
                intr[:2] *= img_scale
                extr = self.dataset.extr_mats[cam_id].copy()
                img_h, img_w = int(self.dataset.img_heights[cam_id] * img_scale), int(self.dataset.img_widths[cam_id] * img_scale)
            elif view_setting.startswith('free'):
                # free view setting
                # frame_num_per_circle = 360
                frame_num_per_circle = 216
                rot_Y = (idx % frame_num_per_circle) / float(frame_num_per_circle) * 2 * np.pi

                extr = visualize_util.calc_free_mv(object_center,
                                                   tar_pos = np.array([0, 0, 2.5]),
                                                   rot_Y = rot_Y,
                                                   rot_X = 0.3 if view_setting.endswith('bird') else 0.,
                                                   global_orient = global_orient if self.opt['test'].get('global_orient', False) else None)
                intr = np.array([[1100, 0, 512], [0, 1100, 512], [0, 0, 1]], np.float32)
                intr[:2] *= img_scale
                img_h = int(1024 * img_scale)
                img_w = int(1024 * img_scale)
            elif view_setting.startswith('front'):
                # front view setting
                extr = visualize_util.calc_free_mv(object_center,
                                                   tar_pos = np.array([0, 0, 2.5]),
                                                   rot_Y = 0.,
                                                   rot_X = 0.3 if view_setting.endswith('bird') else 0.,
                                                   global_orient = global_orient if self.opt['test'].get('global_orient', False) else None)
                intr = np.array([[1100, 0, 512], [0, 1100, 512], [0, 0, 1]], np.float32)
                intr[:2] *= img_scale
                img_h = int(1024 * img_scale)
                img_w = int(1024 * img_scale)
            elif view_setting.startswith('back'):
                # back view setting
                extr = visualize_util.calc_free_mv(object_center,
                                                   tar_pos = np.array([0, 0, 2.5]),
                                                   rot_Y = np.pi,
                                                   rot_X = 0.5 * np.pi / 4. if view_setting.endswith('bird') else 0.,
                                                   global_orient = global_orient if self.opt['test'].get('global_orient', False) else None)
                intr = np.array([[1100, 0, 512], [0, 1100, 512], [0, 0, 1]], np.float32)
                intr[:2] *= img_scale
                img_h = int(1024 * img_scale)
                img_w = int(1024 * img_scale)
            elif view_setting.startswith('moving'):
                # moving camera setting
                extr = visualize_util.calc_free_mv(object_center,
                                                   # tar_pos = np.array([0, 0, 3.0]),
                                                   # rot_Y = -0.3,
                                                   tar_pos = np.array([0, 0, 2.5]),
                                                   rot_Y = 0.,
                                                   rot_X = 0.3 if view_setting.endswith('bird') else 0.,
                                                   global_orient = global_orient if self.opt['test'].get('global_orient', False) else None)
                intr = np.array([[1100, 0, 512], [0, 1100, 512], [0, 0, 1]], np.float32)
                intr[:2] *= img_scale
                img_h = int(1024 * img_scale)
                img_w = int(1024 * img_scale)
            elif view_setting.startswith('cano'):
                cano_center = self.dataset.cano_bounds.mean(0)
                extr = np.identity(4, np.float32)
                extr[:3, 3] = -cano_center
                rot_x = np.identity(4, np.float32)
                rot_x[:3, :3] = cv.Rodrigues(np.array([np.pi, 0, 0], np.float32))[0]
                extr = rot_x @ extr
                f_len = 5000
                extr[2, 3] += f_len / 512
                intr = np.array([[f_len, 0, 512], [0, f_len, 512], [0, 0, 1]], np.float32)
                # item = self.dataset.getitem(idx,
                #                             training = False,
                #                             extr = extr,
                #                             intr = intr,
                #                             img_w = 1024,
                #                             img_h = 1024)
                img_w, img_h = 1024, 1024
                # item['live_smpl_v'] = item['cano_smpl_v']
                # item['cano2live_jnt_mats'] = torch.eye(4, dtype = torch.float32)[None].expand(item['cano2live_jnt_mats'].shape[0], -1, -1)
                # item['live_bounds'] = item['cano_bounds']
            else:
                raise ValueError('Invalid view setting for animation!')

            getitem_func = self.dataset.getitem_fast if hasattr(self.dataset, 'getitem_fast') else self.dataset.getitem
            item = getitem_func(
                idx,
                training = False,
                extr = extr,
                intr = intr,
                img_w = img_w,
                img_h = img_h
            )
            if view_setting == 'cano':
                item['live_smpl_v'] = item['cano_smpl_v']
                item['cano2live_jnt_mats'] = torch.eye(4, dtype = torch.float32)[None].expand(item['cano2live_jnt_mats'].shape[0], -1, -1)
                item['live_bounds'] = item['cano_bounds']
            if test_cameras is not None:
                test_cameras['pose_idx'].append(item['data_idx'])
                test_cameras['intr'].append(item['intr'])
                test_cameras['extr'].append(item['extr'])
                test_cameras['img_w'].append(item['img_w'])
                test_cameras['img_h'].append(item['img_h'])
            items = to_cuda(item, add_batch = False)

            if view_setting.startswith('moving') or view_setting == 'free_moving':
                current_center = items['live_bounds'].cpu().numpy().mean(0)
                delta = current_center - object_center

                object_center[0] += delta[0]
                # object_center[1] += delta[1]
                # object_center[2] += delta[2]

            if log_time:
                time_end.record()
                torch.cuda.synchronize()
                print('Loading data costs %.4f secs' % (time_start.elapsed_time(time_end) / 1000.))
                time_start.record()

            if self.opt['test'].get('render_skeleton', False):
                from utils.visualize_skeletons import construct_skeletons
                skel_vertices, skel_faces = construct_skeletons(item['joints'].cpu().numpy(), item['kin_parent'].cpu().numpy())
                skel_mesh = trimesh.Trimesh(skel_vertices, skel_faces, process = False)

                if geo_renderer is None:
                    geo_renderer = Renderer(item['img_w'], item['img_h'], shader_name = 'phong_geometry', bg_color = (1, 1, 1))
                extr, intr = item['extr'], item['intr']
                geo_renderer.set_camera(extr, intr)
                geo_renderer.set_model(skel_vertices[skel_faces.reshape(-1)], skel_mesh.vertex_normals.astype(np.float32)[skel_faces.reshape(-1)])
                skel_img = geo_renderer.render()[:, :, :3]
                skel_img = (skel_img * 255).astype(np.uint8)
                cv.imwrite(output_dir + '/live_skeleton/%08d.jpg' % item['data_idx'], skel_img)

            if log_time:
                time_end.record()
                torch.cuda.synchronize()
                print('Rendering skeletons costs %.4f secs' % (time_start.elapsed_time(time_end) / 1000.))
                time_start.record()

            if 'smpl_pos_map' not in items:
                self.avatar_net.get_pose_map(items)

            # pca
            if use_pca:
                mask = training_dataset.pos_map_mask
                live_pos_map = items['smpl_pos_map'].permute(1, 2, 0).cpu().numpy()
                front_live_pos_map, back_live_pos_map = np.split(live_pos_map, [3], 2)
                pose_conds = front_live_pos_map[mask]
                new_pose_conds = training_dataset.transform_pca(pose_conds, sigma_pca = float(self.opt['test'].get('sigma_pca', 2.)))
                front_live_pos_map[mask] = new_pose_conds
                live_pos_map = np.concatenate([front_live_pos_map, back_live_pos_map], 2)
                items.update({
                    'smpl_pos_map_pca': torch.from_numpy(live_pos_map).to(config.device).permute(2, 0, 1)
                })

            if log_time:
                time_end.record()
                torch.cuda.synchronize()
                print('Rendering pose conditions costs %.4f secs' % (time_start.elapsed_time(time_end) / 1000.))
                time_start.record()

            if self.opt['test'].get('render_gaussian_only', False):
                prefix = 'gaussian_'
                output = self.avatar_net.render_gaussian_only(items, bg_color = self.bg_color_cuda, use_pca = use_pca)
            else:
                prefix = ''
                output = self.avatar_net.render(items, bg_color = self.bg_color_cuda, use_pca = use_pca)
            if log_time:
                time_end.record()
                torch.cuda.synchronize()
                print('Rendering avatar costs %.4f secs' % (time_start.elapsed_time(time_end) / 1000.))
                time_start.record()

            rgb_map = output['rgb_map']
            rgb_map.clip_(0., 1.)
            rgb_map = (rgb_map * 255).to(torch.uint8).cpu().numpy()
            os.makedirs(output_dir + f'/{prefix}rgb_map', exist_ok = True)
            cv.imwrite(output_dir + f'/{prefix}rgb_map/%08d.jpg' % item['data_idx'], rgb_map)

            if 'mask_map' in output:
                os.makedirs(output_dir + f'/{prefix}mask_map', exist_ok = True)
                mask_map = output['mask_map'][:, :, 0]
                mask_map.clip_(0., 1.)
                mask_map = (mask_map * 255).to(torch.uint8)
                cv.imwrite(output_dir + f'/{prefix}mask_map/%08d.png' % item['data_idx'], mask_map.cpu().numpy())

            if self.opt['test'].get('save_tex_map', False):
                os.makedirs(output_dir + f'/{prefix}cano_tex_map', exist_ok = True)
                cano_tex_map = output['cano_tex_map']
                cano_tex_map.clip_(0., 1.)
                cano_tex_map = (cano_tex_map * 255).to(torch.uint8)
                cv.imwrite(output_dir + f'/{prefix}cano_tex_map/%08d.jpg' % item['data_idx'], cano_tex_map.cpu().numpy())

            if idx == 0 and self.opt['test'].get('save_cano_offset', False):
                rgb = torch.clamp(SH2RGB(output['posed_gaussians']['colors']).view(-1, 3)[..., [2,1,0]], 0., 1.)
                export_pts(
                    filename = output_dir + f'/{prefix}cano_gaussian.ply', 
                    pts = (self.avatar_net.cano_gaussian_model.get_xyz), 
                    color = rgb, 
                    scalars = {'radius': 0.001 * np.ones((rgb.shape[0], 1))}
                )

            if self.opt['test'].get('save_ply', False):
                os.makedirs(output_dir + f'/{prefix}posed_gaussians', exist_ok = True)
                rgb = torch.clamp(SH2RGB(output['posed_gaussians']['colors']).view(-1, 3)[..., [2,1,0]], 0., 1.)
                export_pts(
                    filename = output_dir + f'/{prefix}posed_gaussians/%08d.ply' % item['data_idx'], 
                    pts = output['posed_gaussians']['positions'], 
                    color = rgb, 
                    scalars = {'radius': 0.001 * np.ones((rgb.shape[0], 1))}
                )

            if self.opt['test'].get('save_cano_pts', False):
                os.makedirs(output_dir + f'/{prefix}cano_pts', exist_ok = True)
                rgb = torch.clamp(SH2RGB(output['posed_gaussians']['colors']).view(-1, 3)[..., [2,1,0]], 0., 1.)
                export_pts(
                    filename = output_dir + f'/{prefix}cano_pts/%08d.ply' % item['data_idx'], 
                    pts = output['cano_pts'], 
                    color = rgb, 
                    scalars = {'radius': 0.001 * np.ones((rgb.shape[0], 1))}
                )

            if log_time:
                time_end.record()
                torch.cuda.synchronize()
                print('Saving images costs %.4f secs' % (time_start.elapsed_time(time_end) / 1000.))
                print('Animating one frame costs %.4f secs' % (time_start_all.elapsed_time(time_end) / 1000.))

            torch.cuda.empty_cache()
        
        if save_cameras:
            for k, v in test_cameras.items():
                test_cameras[k] = np.stack(v, axis=0)
            np.savez(output_dir + '/cameras.npz', **test_cameras)

    def save_ckpt(self, path, save_optm = True):
        os.makedirs(path, exist_ok = True)
        net_dict = {
            'epoch_idx': self.epoch_idx,
            'iter_idx': self.iter_idx,
            'avatar_net': self.avatar_net.state_dict(),
        }
        print('Saving networks to ', path + '/net.pt')
        torch.save(net_dict, path + '/net.pt')

        if save_optm:
            optm_dict = {
                'avatar_net': self.optm.state_dict(),
            }
            print('Saving optimizers to ', path + '/optm.pt')
            torch.save(optm_dict, path + '/optm.pt')

    def load_ckpt(self, path, load_optm = True):
        print('Loading networks from ', path + '/net.pt')
        net_dict = torch.load(path + '/net.pt')
        if 'avatar_net' in net_dict:
            self.avatar_net.load_state_dict(net_dict['avatar_net'])
        else:
            print('[WARNING] Cannot find "avatar_net" from the network checkpoint!')
        epoch_idx = net_dict['epoch_idx']
        iter_idx = net_dict['iter_idx']

        if load_optm and os.path.exists(path + '/optm.pt'):
            print('Loading optimizers from ', path + '/optm.pt')
            optm_dict = torch.load(path + '/optm.pt')
            if 'avatar_net' in optm_dict:
                self.optm.load_state_dict(optm_dict['avatar_net'])
            else:
                print('[WARNING] Cannot find "avatar_net" from the optimizer checkpoint!')

        return epoch_idx, iter_idx


if __name__ == '__main__':
    torch.manual_seed(31359)
    np.random.seed(31359)
    # torch.autograd.set_detect_anomaly(True)
    from argparse import ArgumentParser

    arg_parser = ArgumentParser()
    arg_parser.add_argument('-c', '--config_path', type = str, help = 'Configuration file path.')
    arg_parser.add_argument('-m', '--mode', type = str, help = 'Running mode.', default = 'train')
    arg_parser.add_argument('--exp_name', type = str, help = 'Pretrain mode.', default = None)
    arg_parser.add_argument('--exp_dir', type = str, help = 'Experiment directory.', default = None)
    args = arg_parser.parse_args()

    config.load_global_opt(args.config_path)
    if args.mode is not None:
        config.opt['mode'] = args.mode
    if args.exp_name is not None:
        config.opt['exp_name'] = args.exp_name
    else:
        config.opt['exp_name'] = os.path.splitext(os.path.basename(args.config_path))[0]
    if args.mode == 'train':
        if args.exp_dir is None:
            config.opt['train']['net_ckpt_dir'] = config.opt['train']['net_ckpt_dir'] + '/' + config.opt['exp_name'] + '/' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            # save opt
            os.makedirs(config.opt['train']['net_ckpt_dir'], exist_ok = True)
            with open(os.path.join(config.opt['train']['net_ckpt_dir'], 'opt.yaml'), 'w') as f:
                yaml.dump(config.opt, f, indent = 4)
        else:
            config.opt['train']['net_ckpt_dir'] = args.exp_dir
    else:
        if os.path.exists(config.opt['test']['prev_ckpt']):
            config.opt['train']['net_ckpt_dir'] = config.opt['test']['prev_ckpt']
        elif args.exp_dir is not None:
            config.opt['train']['net_ckpt_dir'] = args.exp_dir
        else:
            raise FileNotFoundError('Cannot find checkpoint!')

    trainer = AvatarTrainer(config.opt)
    if config.opt['mode'] == 'train':
        if not safe_exists(config.opt['train']['net_ckpt_dir'] + '/pretrained') \
                and not safe_exists(config.opt['train']['pretrained_dir'])\
                and not safe_exists(config.opt['train']['prev_ckpt']):
            trainer.pretrain()
        trainer.train()
    elif config.opt['mode'] == 'test':
        trainer.test()
    else:
        raise NotImplementedError('Invalid running mode!')
