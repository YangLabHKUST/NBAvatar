import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch3d.ops
import pytorch3d.transforms
import cv2 as cv

import config
from network.mlp import MLPLinear
from network.styleunet.dual_styleunet import StyleUNet
from gaussians.gaussian_model import GaussianModelTorch
from gaussians.gaussian_renderer import render3
from network.pe import PositionalEncodingEmbedder


class AvatarNet(nn.Module):
    def __init__(self, opt):
        super(AvatarNet, self).__init__()
        self.opt = opt

        self.random_style = opt.get('random_style', False)
        self.with_viewdirs = opt.get('with_viewdirs', True)
        self.input_size = opt.get('input_size', 512)
        self.geo_bw_size = opt.get('geo_bw_size', 64)
        self.rgb_bw_size = opt.get('rgb_bw_size', 128)
        self.geo_middle_size = opt.get('geo_middle_size', 16)
        self.rgb_middle_size = opt.get('rgb_middle_size', 16)
        self.geo_feature_size = opt.get('geo_feature_size', 128)
        self.rgb_feature_size = opt.get('rgb_feature_size', 256)
        self.feature_dim = opt.get('feature_dim', 32)
        self.geo_inp_size = opt.get('geo_inp_size', 1024)
        self.rgb_inp_size = opt.get('rgb_inp_size', 1024)
        self.output_size = opt.get('output_size', 1024)
        self.num_basis = opt.get('num_basis', 20)
        self.add_global = opt.get('add_global', False)
        self.pred_color_offset = opt.get('pred_color_offset', True)
        self.multires_pe = opt.get('multires_pe', 0)
        self.max_sh_degree = opt.get('max_sh_degree', 0)
        self.opt_cano_gs = opt.get('opt_cano_gs', True)
        self.mlp_inter = opt.get('mlp_inter', [128, 128])
        self.fix_cano_xyz = opt.get('fix_cano_xyz', False)
        self.opt_xyz_offset = opt.get('opt_xyz_offset', False)
        # self.pre_mask_feat = opt.get('pre_mask_feat', False)

        if self.multires_pe > 0:
            self.pe = PositionalEncodingEmbedder(self.multires_pe)
            pe_ch = self.pe.out_dim
        else:
            self.pe = None
            pe_ch = 3

        # init canonical gausssian model
        cano_smpl_map = cv.imread(config.opt['train']['data']['data_dir'] + '/smpl_pos_map/cano_smpl_pos_map.exr', cv.IMREAD_UNCHANGED)
        self.cano_smpl_map = torch.from_numpy(cano_smpl_map).to(torch.float32).to(config.device)
        self.cano_smpl_mask = torch.linalg.norm(self.cano_smpl_map, dim = -1) > 0.
        self.init_points = self.cano_smpl_map[self.cano_smpl_mask]
        self.lbs = torch.from_numpy(np.load(config.opt['train']['data']['data_dir'] + '/smpl_pos_map/init_pts_lbs.npy')).to(torch.float32).to(config.device)
        self.cano_gaussian_model = GaussianModelTorch(
            points = self.init_points, 
            colors = 0.5 * torch.ones_like(self.init_points), 
            sh_degree = self.max_sh_degree, 
            spatial_lr_scale = 2.5,
            opt_xyz_offset = self.opt_xyz_offset
        )
        if not self.opt_cano_gs:
            for p in self.cano_gaussian_model.parameters():
                p.requires_grad = False
        if self.fix_cano_xyz:
            print('Fix cano xyz')
            self.cano_gaussian_model._xyz.requires_grad = False
        self.gs_sh_dim = self.cano_gaussian_model.get_sh_dim
        self.gs_out_dim = self.cano_gaussian_model.get_out_dim

        if self.with_viewdirs:
            cano_nml_map = cv.imread(config.opt['train']['data']['data_dir'] + '/smpl_pos_map/cano_smpl_nml_map.exr', cv.IMREAD_UNCHANGED)
            self.cano_nml_map = torch.from_numpy(cano_nml_map).to(torch.float32).to(config.device)
            self.cano_nmls = self.cano_nml_map[self.cano_smpl_mask]
            self.viewdir_net = nn.Sequential(
                nn.Conv2d(1, 8, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Conv2d(8, 16, 4, 2, 1)
            )
            self.view_feature = 16
        else:
            self.viewdir_net = None
            self.view_feature = 0

        self.color_bw_net = StyleUNet(
            inp_size = self.input_size, 
            inp_ch = 3, 
            out_ch = self.num_basis, 
            out_size = self.geo_bw_size, 
            style_dim = 512, 
            n_mlp = 2, 
            middle_size=self.geo_middle_size
        )
        self.color_out_net = MLPLinear(
            self.feature_dim + pe_ch + self.view_feature, 
            self.gs_out_dim, 
            self.mlp_inter, 
            nlactv = nn.LeakyReLU(0.2, inplace = True), 
            last_op = None, 
            last_std = 0.001
        )
        self.other_out_net = MLPLinear(
            self.feature_dim + pe_ch, 8, 
            self.mlp_inter, 
            nlactv = nn.LeakyReLU(0.2, inplace = True), 
            last_op = None, 
            last_std = 0.001
        )
        if self.add_global:
            self.color_basis = nn.Parameter(0.01 * torch.randn(self.num_basis + 1, self.feature_dim, self.rgb_feature_size, self.rgb_feature_size))
        else:
            self.color_basis = nn.Parameter(0.01 * torch.randn(self.num_basis, self.feature_dim, self.rgb_feature_size, self.rgb_feature_size))
        print('Using color_basis with shape:', self.color_basis.shape)

        self.geom_bw_net = StyleUNet(
            inp_size = self.input_size, 
            inp_ch = 3, 
            out_ch = self.num_basis, 
            out_size = self.rgb_bw_size, 
            style_dim = 512, 
            n_mlp = 2, 
            middle_size=self.rgb_middle_size
        )
        self.geom_out_net = MLPLinear(
            self.feature_dim + pe_ch, 
            3, 
            self.mlp_inter, 
            nlactv = nn.LeakyReLU(0.2, inplace = True), 
            last_op = None, 
            last_std = 0.001
        )
        if self.add_global:
            self.geom_basis = nn.Parameter(0.01 * torch.randn(self.num_basis + 1, self.feature_dim, self.geo_feature_size, self.geo_feature_size))
        else:
            self.geom_basis = nn.Parameter(0.01 * torch.randn(self.num_basis, self.feature_dim, self.geo_feature_size, self.geo_feature_size))
        print('Using geom_basis with shape:', self.geom_basis.shape)

        self.color_style = torch.ones([1, self.color_bw_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.color_bw_net.style_dim)
        self.geom_style = torch.ones([1, self.geom_bw_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.geom_bw_net.style_dim)

    def generate_mean_hands(self):
        raise NotImplementedError
        # print('# Generating mean hands ...')
        import glob
        # get hand mask
        lbs_argmax = self.lbs.argmax(1)
        self.hand_mask = lbs_argmax == 20
        self.hand_mask = torch.logical_or(self.hand_mask, lbs_argmax == 21)
        self.hand_mask = torch.logical_or(self.hand_mask, lbs_argmax >= 25)

        pose_map_paths = sorted(glob.glob(config.opt['train']['data']['data_dir'] + '/smpl_pos_map/%08d.exr' % config.opt['test']['fix_hand_id']))
        smpl_pos_map = cv.imread(pose_map_paths[0], cv.IMREAD_UNCHANGED)
        pos_map_size = smpl_pos_map.shape[1] // 2
        smpl_pos_map = np.concatenate([smpl_pos_map[:, :pos_map_size], smpl_pos_map[:, pos_map_size:]], 2)
        smpl_pos_map = smpl_pos_map.transpose((2, 0, 1))
        pose_map = torch.from_numpy(smpl_pos_map).to(torch.float32).to(config.device)
        pose_map = pose_map[:3]

        cano_pts = self.get_positions(pose_map)
        opacity, scales, rotations = self.get_others(pose_map)
        colors, color_map = self.get_colors(pose_map)

        self.hand_positions = cano_pts#[self.hand_mask]
        self.hand_opacity = opacity#[self.hand_mask]
        self.hand_scales = scales#[self.hand_mask]
        self.hand_rotations = rotations#[self.hand_mask]
        self.hand_colors = colors#[self.hand_mask]

        # # debug
        # hand_pts = trimesh.PointCloud(self.hand_positions.detach().cpu().numpy())
        # hand_pts.export('./debug/hand_template.obj')
        # exit(1)

    def transform_cano2live(self, gaussian_vals, items):
        pt_mats = torch.einsum('nj,jxy->nxy', self.lbs, items['cano2live_jnt_mats'])
        gaussian_vals['positions'] = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], gaussian_vals['positions']) + pt_mats[..., :3, 3]
        rot_mats = pytorch3d.transforms.quaternion_to_matrix(gaussian_vals['rotations'])
        rot_mats = torch.einsum('nxy,nyz->nxz', pt_mats[..., :3, :3], rot_mats)
        gaussian_vals['rotations'] = pytorch3d.transforms.matrix_to_quaternion(rot_mats)

        return gaussian_vals

    def get_geometry(self, pose_map, return_delta = False, return_map = False, faster = False):
        geo_bw, _ = self.geom_bw_net([self.geom_style], pose_map[None], randomize_noise = False) # (1, N, H, W)
        if faster:
            geo_bw = F.interpolate(geo_bw, (self.geo_inp_size, self.geo_inp_size), mode = 'bilinear', align_corners = False)
            geom_basis = F.interpolate(self.geom_basis, (self.geo_inp_size, self.geo_inp_size), mode = 'bilinear', align_corners = False)
            if self.add_global:
                bw_feats = geom_basis[:1] + torch.sum(geo_bw.permute(1, 0, 2, 3) * geom_basis[1:], dim = 0, keepdim = True) # (1, C, H, W)
            else:
                bw_feats = torch.sum(geo_bw.permute(1, 0, 2, 3) * geom_basis, dim = 0, keepdim = True) # (1, C, H, W)
        else:
            if self.geo_bw_size != self.geo_feature_size:
                geo_bw = F.interpolate(geo_bw, (self.geo_feature_size, self.geo_feature_size), mode = 'bilinear', align_corners = False)
            # self.geom_basis: (N, C, H, W)
            if self.add_global:
                bw_feats = self.geom_basis[:1] + torch.sum(geo_bw.permute(1, 0, 2, 3) * self.geom_basis[1:], dim = 0, keepdim = True) # (1, C, H, W)
            else:
                bw_feats = torch.sum(geo_bw.permute(1, 0, 2, 3) * self.geom_basis, dim = 0, keepdim = True) # (1, C, H, W)
            bw_feats = F.interpolate(bw_feats, (self.geo_inp_size, self.geo_inp_size), mode = 'bilinear', align_corners = False)
        bw_feats = bw_feats[0].permute(1, 2, 0)[self.cano_smpl_mask]
        pos = self.pe(self.init_points)
        geom_out = self.geom_out_net(torch.cat([pos, bw_feats], 1))

        position_map = geom_out[..., :3]
        delta_position = 0.05 * position_map

        if return_delta:
            positions = delta_position
        else:
            if self.opt_xyz_offset:
                positions = delta_position + self.cano_gaussian_model.get_xyz + torch.tanh(self.xyz_offset) * 0.008 # trick
            else:
                positions = delta_position + self.cano_gaussian_model.get_xyz

        if return_map:
            map_dict = {
                'position_map': position_map,
            }
            return positions, map_dict, geo_bw
        else:
            return positions, geo_bw

    # def get_others(self, pose_map):
    #     other_map, _ = self.other_net([self.other_style], pose_map[None], randomize_noise = False)
    #     front_map, back_map = torch.split(other_map, [8, 8], 1)
    #     other_map = torch.cat([front_map, back_map], 3)[0].permute(1, 2, 0)
    #     others = other_map[self.cano_smpl_mask]  # (N, 8)
    #     opacity, scales, rotations = torch.split(others, [1, 3, 4], 1)
    #     opacity = self.cano_gaussian_model.opacity_activation(opacity + self.cano_gaussian_model.get_opacity_raw)
    #     scales = self.cano_gaussian_model.scaling_activation(scales + self.cano_gaussian_model.get_scaling_raw)
    #     rotations = self.cano_gaussian_model.rotation_activation(rotations + self.cano_gaussian_model.get_rotation_raw)

    #     return opacity, scales, rotations

    def get_colors(self, pose_map, viewdirs = None, return_delta = False, return_map = False, faster = False):
        rgb_bw, _ = self.color_bw_net([self.color_style], pose_map[None], randomize_noise = False)
        if faster: # faster actully slow sad:(
            rgb_bw = F.interpolate(rgb_bw, (self.rgb_inp_size, self.rgb_inp_size), mode = 'bilinear', align_corners = False)
            color_basis = F.interpolate(self.color_basis, (self.rgb_inp_size, self.rgb_inp_size), mode = 'bilinear', align_corners = False)
            if self.add_global:
                bw_feats = color_basis[:1] + torch.sum(rgb_bw.permute(1, 0, 2, 3) * color_basis[1:], dim = 0, keepdim = True) # (1, C, H, W)
            else:
                bw_feats = torch.sum(rgb_bw.permute(1, 0, 2, 3) * color_basis, dim = 0, keepdim = True) # (1, C, H, W)
        else:
            if self.rgb_bw_size != self.rgb_feature_size:
                rgb_bw = F.interpolate(rgb_bw, (self.rgb_feature_size, self.rgb_feature_size), mode = 'bilinear', align_corners = False)
            # self.color_basis: (N, C, H, W)
            if self.add_global:
                bw_feats = self.color_basis[:1] + torch.sum(rgb_bw.permute(1, 0, 2, 3) * self.color_basis[1:], dim = 0, keepdim = True) # (1, C, H, W)
            else:
                bw_feats = torch.sum(rgb_bw.permute(1, 0, 2, 3) * self.color_basis, dim = 0, keepdim = True) # (1, C, H, W) 
            bw_feats = F.interpolate(bw_feats, (self.rgb_inp_size, self.rgb_inp_size), mode = 'bilinear', align_corners = False)
        bw_feats = bw_feats[0].permute(1, 2, 0)[self.cano_smpl_mask]
        pos = self.pe(self.init_points)
        if viewdirs is not None:
            viewdirs = F.interpolate(viewdirs, (self.rgb_inp_size, self.rgb_inp_size), mode = 'bilinear', align_corners = False)
            viewdirs = viewdirs[0].permute(1, 2, 0)[self.cano_smpl_mask]
            rgb_inp = torch.cat([bw_feats, pos, viewdirs], 1)
            color_out = self.color_out_net(rgb_inp)
        else:
            color_out = None
        other_inp = torch.cat([bw_feats, pos], 1)
        other_out = self.other_out_net(other_inp)

        if self.pred_color_offset:
            if color_out is None:
                delta_color = torch.zeros_like(self.cano_gaussian_model.get_features)
            else:
                delta_color = color_out.view(-1, self.gs_sh_dim, 3) * 0.1
            if return_delta:
                colors = delta_color
            else:
                colors = self.cano_gaussian_model.get_features + delta_color
        else:
            colors = color_out.view(-1, self.gs_sh_dim, 3)

        delta_opacity = other_out[..., 0:1]
        delta_scales = other_out[..., 1:4]
        delta_rotations = other_out[..., 4:8]
        if return_delta:
            opacity = delta_opacity
            scales = delta_scales
            rotations = delta_rotations
        else:
            opacity = self.cano_gaussian_model.opacity_activation(delta_opacity + self.cano_gaussian_model.get_opacity_raw)
            scales = self.cano_gaussian_model.scaling_activation(delta_scales + self.cano_gaussian_model.get_scaling_raw)
            rotations = self.cano_gaussian_model.rotation_activation(delta_rotations + self.cano_gaussian_model.get_rotation_raw)

        if return_map:
            map_dict = {
                'color_map': color_out,
                'other_map': other_out,
            }
            return opacity, scales, rotations, colors, map_dict, rgb_bw
        else:
            return opacity, scales, rotations, colors, rgb_bw    

    def get_viewdir_feat(self, items):
        with torch.no_grad():
            pt_mats = torch.einsum('nj,jxy->nxy', self.lbs, items['cano2live_jnt_mats'])
            live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.init_points) + pt_mats[..., :3, 3]
            live_nmls = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.cano_nmls)
            cam_pos = -torch.matmul(torch.linalg.inv(items['extr'][:3, :3]), items['extr'][:3, 3])
            viewdirs = F.normalize(cam_pos[None] - live_pts, dim = -1, eps = 1e-3)
            if self.training:
                viewdirs += torch.randn(*viewdirs.shape).to(viewdirs) * 0.1
            viewdirs = F.normalize(viewdirs, dim = -1, eps = 1e-3)
            viewdirs = (live_nmls * viewdirs).sum(-1)

            viewdirs_map = torch.zeros(*self.cano_nml_map.shape[:2]).to(viewdirs)
            viewdirs_map[self.cano_smpl_mask] = viewdirs

            viewdirs_map = viewdirs_map[None, None]
            viewdirs_map = F.interpolate(viewdirs_map, None, 0.5, 'nearest')

        viewdirs = self.opt.get('weight_viewdirs', 1.) * self.viewdir_net(viewdirs_map)
        return viewdirs

    def get_pose_map(self, items):
        pt_mats = torch.einsum('nj,jxy->nxy', self.lbs, items['cano2live_jnt_mats_woRoot'])
        live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.init_points) + pt_mats[..., :3, 3]
        live_pos_map = torch.zeros_like(self.cano_smpl_map)
        live_pos_map[self.cano_smpl_mask] = live_pts
        live_pos_map = F.interpolate(live_pos_map.permute(2, 0, 1)[None], None, [0.5, 0.5], mode = 'nearest')[0]
        # live_pos_map = torch.cat(torch.split(live_pos_map, [512, 512], 2), 0)
        items.update({
            'smpl_pos_map': live_pos_map
        })
        return live_pos_map
    
    def render_gaussian_only(self, items, bg_color = (0., 0., 0), use_pca = False, use_vae = False, sh_degree = None):
        if isinstance(bg_color, np.ndarray) or isinstance(bg_color, list) or isinstance(bg_color, tuple):
            bg_color = torch.from_numpy(np.asarray(bg_color)).to(torch.float32).to(config.device)
        else:
            bg_color = bg_color.to(config.device)

        if self.opt_xyz_offset:
            positions = self.cano_gaussian_model.get_xyz + torch.tanh(self.xyz_offset) * 0.008 # a trick
        else:
            positions = self.cano_gaussian_model.get_xyz
        gaussian_vals = {
            'positions': positions,
            'opacity': self.cano_gaussian_model.get_opacity,
            'scales': self.cano_gaussian_model.get_scaling,
            'rotations': self.cano_gaussian_model.get_rotation,
            'colors': self.cano_gaussian_model.get_features,
            'sh_degree': self.max_sh_degree if sh_degree is None else sh_degree
        }

        nonrigid_offset = gaussian_vals['positions'] - self.init_points

        gaussian_vals = self.transform_cano2live(gaussian_vals, items)

        render_ret = render3(
            gaussian_vals,
            bg_color,
            items['extr'],
            items['intr'],
            items['img_w'],
            items['img_h']
        )
        if not self.training:
            rgb_map = torch.clamp(render_ret['render'], 0., 1.)
        else:
            rgb_map = render_ret['render']
        dpt_map = render_ret['depth']
        mask_map = render_ret['mask']

        ret = {
            'rgb_map': rgb_map,
            'dpt_map': dpt_map,
            'mask_map': mask_map,
            'offset': nonrigid_offset,
            'posed_gaussians': gaussian_vals,
        }

        return ret

    def render(self, items, bg_color = (0., 0., 0.), use_pca = False, use_vae = False, sh_degree = None):
        """
        Note that no batch index in items.
        """
        if isinstance(bg_color, np.ndarray) or isinstance(bg_color, list) or isinstance(bg_color, tuple):
            bg_color = torch.from_numpy(np.asarray(bg_color)).to(torch.float32).to(config.device)
        else:
            bg_color = bg_color.to(config.device)
        pose_map = items['smpl_pos_map'][:3]
        assert not (use_pca and use_vae), "Cannot use both PCA and VAE!"
        if use_pca:
            pose_map = items['smpl_pos_map_pca'][:3]
        if use_vae:
            pose_map = items['smpl_pos_map_vae'][:3]

        cano_pts, pos_map, geo_bw = self.get_geometry(pose_map, return_map = True, return_delta = False)
        # if not self.training:
        # scales = torch.clip(scales, 0., 0.03)
        if self.with_viewdirs:
            assert self.max_sh_degree == 0, 'max_sh_degree must be 0 when with_viewdirs is True'
            viewdirs = self.get_viewdir_feat(items)
        else:
            viewdirs = None
        opacity, scales, rotations, colors, color_map, rgb_bw = self.get_colors(pose_map, viewdirs, return_map = True, return_delta = False)

        if not self.training and config.opt['test'].get('fix_hand', False) and config.opt['mode'] == 'test':
            # print('# fuse hands ...')
            import utils.geo_util as geo_util
            cano_xyz = self.init_points
            wl = torch.sigmoid(2.5 * (geo_util.normalize_vert_bbox(items['left_cano_mano_v'], attris = cano_xyz, dim = 0, per_axis = True)[..., 0:1] + 2.0))
            wr = torch.sigmoid(-2.5 * (geo_util.normalize_vert_bbox(items['right_cano_mano_v'], attris = cano_xyz, dim = 0, per_axis = True)[..., 0:1] - 2.0))
            wl[cano_xyz[..., 1] < items['cano_smpl_center'][1]] = 0.
            wr[cano_xyz[..., 1] < items['cano_smpl_center'][1]] = 0.

            s = torch.maximum(wl + wr, torch.ones_like(wl))
            wl, wr = wl / s, wr / s

            w = wl + wr
            cano_pts = w * self.hand_positions + (1.0 - w) * cano_pts
            opacity = w * self.hand_opacity + (1.0 - w) * opacity
            scales = w * self.hand_scales + (1.0 - w) * scales
            rotations = w * self.hand_rotations + (1.0 - w) * rotations
            # colors = w * self.hand_colors + (1.0 - w) * colors

        gaussian_vals = {
            'positions': cano_pts,
            'opacity': opacity,
            'scales': scales,
            'rotations': rotations,
            'colors': colors,
            'sh_degree': self.max_sh_degree if sh_degree is None else sh_degree
        }

        nonrigid_offset = gaussian_vals['positions'] - self.cano_gaussian_model.get_xyz.detach()

        gaussian_vals = self.transform_cano2live(gaussian_vals, items)

        render_ret = render3(
            gaussian_vals,
            bg_color,
            items['extr'],
            items['intr'],
            items['img_w'],
            items['img_h']
        )
        if not self.training:
            rgb_map = torch.clamp(render_ret['render'], 0., 1.)
        else:
            rgb_map = render_ret['render']
        dpt_map = render_ret['depth']
        mask_map = render_ret['mask']

        ret = {
            'rgb_map': rgb_map,
            'dpt_map': dpt_map,
            'mask_map': mask_map,
            'offset': nonrigid_offset,
            'posed_gaussians': gaussian_vals,
            'geo_bw': geo_bw,
            'rgb_bw': rgb_bw,
        }

        if self.opt_cano_gs and not self.fix_cano_xyz:
            cano_offset = self.cano_gaussian_model.get_xyz - self.init_points
            ret.update({
                'cano_offset': cano_offset,
            })

        if not self.training:
            ret.update({
                'cano_tex_map': color_map['color_map'],
                'position_map': pos_map['position_map'],
                'other_map': color_map['other_map'],
            })

        return ret
