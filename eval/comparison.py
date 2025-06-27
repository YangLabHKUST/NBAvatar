# To compute FID, first install pytorch_fid
# pip install pytorch-fid

import os
import argparse
import cv2 as cv
from tqdm import tqdm
import shutil
import json

from eval.score import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    metrics = Metrics()
    metrics_dict = {}
    parser.add_argument('--preds_dir', type = str, required = True)
    parser.add_argument('--gt_dir', type = str, required = True)
    parser.add_argument('--mask_dir', type = str, required = True)
    parser.add_argument('--frame_list', type = int, nargs = 3, default = [0, 500, 1])
    parser.add_argument('--actorshq', action = 'store_true', default = False)
    args = parser.parse_args()

    # shutil.rmtree('./tmp_quant')
    os.makedirs('./tmp_quant/preds', exist_ok = True)
    os.makedirs('./tmp_quant/gt', exist_ok = True)
    
    frame_list = list(range(args.frame_list[0], args.frame_list[1], args.frame_list[2]))
    if args.actorshq:
        images_list = [f for f in os.listdir(args.gt_dir) if f.endswith('.jpg')]
        images_list.sort()
        images_list = [os.path.join(args.gt_dir, f) for f in images_list]
        masks_list = [f for f in os.listdir(args.mask_dir) if f.endswith('.png')]
        masks_list.sort()
        masks_list = [os.path.join(args.mask_dir, f) for f in masks_list]

    for frame_id in tqdm(frame_list):
        if args.actorshq:
            if os.path.exists(args.preds_dir + '/%08d.jpg' % frame_id):
                pred_img = (cv.imread(args.preds_dir + '/%08d.jpg' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
            elif os.path.exists(args.preds_dir + '/%08d.png' % frame_id):
                pred_img = (cv.imread(args.preds_dir + '/%08d.png' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
            else:
                raise ValueError('No image found for frame %d' % frame_id)
            gt_img = (cv.imread(images_list[frame_id], cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
            mask_img = cv.imread(masks_list[frame_id], cv.IMREAD_UNCHANGED) > 128
        else:
            if not (os.path.exists(args.gt_dir + '/%08d.jpg' % frame_id) or os.path.exists(args.gt_dir + '/%08d.png' % frame_id)) and \
                not (os.path.exists(args.mask_dir + '/%08d.jpg' % frame_id) or os.path.exists(args.mask_dir + '/%08d.png' % frame_id)):
                continue
            if os.path.exists(args.preds_dir + '/%08d.jpg' % frame_id):
                pred_img = (cv.imread(args.preds_dir + '/%08d.jpg' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
            elif os.path.exists(args.preds_dir + '/%08d.png' % frame_id):
                pred_img = (cv.imread(args.preds_dir + '/%08d.png' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
            else:
                raise ValueError('No image found for frame %d' % frame_id)
            if os.path.exists(args.gt_dir + '/%08d.jpg' % frame_id):
                gt_img = (cv.imread(args.gt_dir + '/%08d.jpg' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
            elif os.path.exists(args.gt_dir + '/%08d.png' % frame_id):
                gt_img = (cv.imread(args.gt_dir + '/%08d.png' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
            else:
                raise ValueError('No image found for frame %d' % frame_id)
            if os.path.exists(args.mask_dir + '/%08d.jpg' % frame_id):
                mask_img = cv.imread(args.mask_dir + '/%08d.jpg' % frame_id, cv.IMREAD_UNCHANGED) > 128
            elif os.path.exists(args.mask_dir + '/%08d.png' % frame_id):
                mask_img = cv.imread(args.mask_dir + '/%08d.png' % frame_id, cv.IMREAD_UNCHANGED) > 128
            else:
                raise ValueError('No image found for frame %d' % frame_id)
        
        gt_img[~mask_img] = 1.

        pred_img_cropped, gt_img_cropped = \
            crop_image(
                mask_img,
                512,
                pred_img,
                gt_img
            )

        cv.imwrite('./tmp_quant/preds/%08d.png' % frame_id, (pred_img_cropped * 255).astype(np.uint8))
        cv.imwrite('./tmp_quant/gt/%08d.png' % frame_id, (gt_img_cropped * 255).astype(np.uint8))

        psnr = compute_psnr(pred_img, gt_img)
        ssim = compute_ssim(pred_img, gt_img)
        lpips = compute_lpips(pred_img_cropped, gt_img_cropped)

        metrics_dict[f'{frame_id:08d}'] = {}
        metrics_dict[f'{frame_id:08d}']['psnr'] = float(psnr)
        metrics_dict[f'{frame_id:08d}']['ssim'] = float(ssim)
        metrics_dict[f'{frame_id:08d}']['lpips'] = float(lpips)

        metrics.psnr += psnr
        metrics.ssim += ssim
        metrics.lpips += lpips
        metrics.count += 1

    print('Ours metrics: ', metrics)
    metrics_dict['psnr'] = float(metrics.psnr / metrics.count)
    metrics_dict['ssim'] = float(metrics.ssim / metrics.count)
    metrics_dict['lpips'] = float(metrics.lpips / metrics.count)
    
    with open(os.path.dirname(args.preds_dir) + '/metrics.json', 'w', encoding = 'utf-8') as f:
        json.dump(metrics_dict, f, indent = 4, ensure_ascii = False, sort_keys = True)

    print('--- Ours ---')
    os.system('python -m pytorch_fid --device cuda {} {}'.format('./tmp_quant/preds', './tmp_quant/gt'))

    shutil.rmtree('./tmp_quant')
