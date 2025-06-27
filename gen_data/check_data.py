import os
import signal
from os.path import join
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import json


exit_flag = False
exit_event = None


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}


def check_image_integrity(file_path):
    """检查图片是否可读且完整"""
    try:
        # 尝试打开验证
        with Image.open(file_path) as img:
            img.verify()  # 验证文件完整性
        # 额外校验JPEG文件结尾
        if file_path.lower().endswith(('.jpg', '.jpeg')):
            with open(file_path, 'rb') as f:
                if f.read()[-2:] != b'\xff\xd9':
                    return False, "JPEG格式不完整"
        return True, "OK"
    except Exception as e:
        return False, str(e)
    

def process_folder(folder_path):
    """处理单个文件夹，返回统计信息和问题文件"""
    valid_images = []
    invalid_images = []
    
    for root, _, files in os.walk(folder_path):
        if exit_event.is_set():
            print(f"Received signal to exit, skipping {folder_path}")
            break
        
        file_count = len(files)
        with tqdm(total=file_count, desc=f"Processing {folder_path}", unit='files', leave=False) as pbar:
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                
                if ext in IMAGE_EXTENSIONS:
                    is_valid, msg = check_image_integrity(file_path)
                    if is_valid:
                        valid_images.append(file_path)
                    else:
                        invalid_images.append((file_path, msg))
                pbar.update(1)
        
    return {
        'folder': folder_path,
        'total': len(valid_images) + len(invalid_images),
        'valid': len(valid_images),
        'invalid': len(invalid_images),
        'valid_images': [os.path.basename(img) for img in valid_images],
        'invalid_images': [os.path.basename(img) for img in invalid_images]
    }


def signal_handler(sig, frame):
    global exit_flag
    print("Received signal to exit")
    exit_flag = True
    exit_event.set()


def main():
    global exit_event

    parser = argparse.ArgumentParser(description='图片完整性检查工具')
    parser.add_argument('--data_path', type=str, help='要检查的文件夹路径')
    parser.add_argument('--images_dir', type=str, default='rgbs', help='图片文件夹路径')
    parser.add_argument('--masks_dir', type=str, default='masks', help='mask文件夹路径')
    parser.add_argument('-p', '--processes', type=int, default=4, help='并行进程数')
    args = parser.parse_args()

    exit_event = multiprocessing.Event()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 多进程处理
    folders = sorted(os.listdir(join(args.data_path, args.images_dir)))
    img_results = []
    with ProcessPoolExecutor(max_workers=args.processes) as executor:
        futures = [executor.submit(process_folder, join(args.data_path, args.images_dir, folder)) for folder in folders]
        with tqdm(total=len(futures), desc="Processing images folders", unit='folders', leave=False) as pbar:
            for future in as_completed(futures):
                if exit_flag:
                    for future in futures:
                        future.cancel()
                    break
                img_results.append(future.result())
                pbar.update(1)

    folders = sorted(os.listdir(join(args.data_path, args.masks_dir)))
    mask_results = []
    with ProcessPoolExecutor(max_workers=args.processes) as executor:
        futures = [executor.submit(process_folder, join(args.data_path, args.masks_dir, folder)) for folder in folders]
        with tqdm(total=len(futures), desc="Processing masks folders", unit='folders', leave=False) as pbar:
            for future in as_completed(futures):
                if exit_flag:
                    for future in futures:
                        future.cancel()
                    break
                mask_results.append(future.result())
                pbar.update(1)

    # 统计分析
    valid_images = defaultdict(set)
    invalid_images = defaultdict(set)
    
    for result in img_results:
        folder = os.path.basename(result['folder'])
        total = result['total']
        valid = result['valid']
        invalid = result['invalid']
        valid_image = sorted([x.replace(folder + '_' + 'rgb', '') for x in result['valid_images']])
        invalid_image = sorted([x.replace(folder + '_' + 'rgb', '') for x in result['invalid_images']])
        
        valid_images[folder] = set(valid_image)
        invalid_images[folder] = set(invalid_image)
        
        print(f"Image文件夹: {folder}")
        print(f"  总文件: {total:,}  有效: {valid:,}  无效: {invalid:,}")

    valid_masks = defaultdict(set)
    invalid_masks = defaultdict(set)
    for result in mask_results:
        folder = os.path.basename(result['folder'])
        total = result['total']
        valid = result['valid']
        invalid = result['invalid']
        
        valid_mask = sorted([x.replace(folder + '_' + 'mask', '') for x in result['valid_images']])
        invalid_mask = sorted([x.replace(folder + '_' + 'mask', '') for x in result['invalid_images']])
        
        valid_masks[folder] = set(valid_mask)
        invalid_masks[folder] = set(invalid_mask)
        
        print(f"Mask文件夹: {folder}")
        print(f"  总文件: {total:,}  有效: {valid:,}  无效: {invalid:,}")
        
    # # 找出数量不一致的文件夹
    # counts = list(valid_counts.values())
    # if len(set(counts)) > 1:
    #     print("警告: 发现图片数量不一致的文件夹!")
    #     print(f"数量分布: {dict(valid_counts)}")
    #     print("建议检查以下文件夹:")
    #     for folder, cnt in valid_counts.items():
    #         if cnt != counts[0]:
    #             print(f"→ {folder}: {cnt} 张 ({'最低' if cnt == min(counts) else '最高' if cnt == max(counts) else ''})")
    # else:
    #     print("所有文件夹图片数量一致")

    # import ipdb; ipdb.set_trace()

    # 寻找所有valid images的交集
    valid_images_set = sorted(list(set.intersection(*valid_images.values())))
    valid_masks_set = sorted(list(set.intersection(*valid_masks.values())))
    with open(os.path.join(args.data_path, 'valid_images.json'), 'w') as f:
        json.dump(valid_images_set, f, indent=4)
    with open(os.path.join(args.data_path, 'valid_masks.json'), 'w') as f:
        json.dump(valid_masks_set, f, indent=4)
    valid_images_set = set([x.strip('.jpg') for x in valid_images_set])
    valid_masks_set = set([x.strip('.png') for x in valid_masks_set])
    valid_set = valid_images_set & valid_masks_set
    with open(os.path.join(args.data_path, 'valid_set.json'), 'w') as f:
        json.dump(sorted(list(valid_set)), f, indent=4)

    # 寻找所有invalid images的并集
    invalid_images_set = set.union(*invalid_images.values())
    invalid_masks_set = set.union(*invalid_masks.values())
    with open(os.path.join(args.data_path, 'invalid_images.json'), 'w') as f:
        json.dump(sorted(list(invalid_images_set)), f, indent=4)
    with open(os.path.join(args.data_path, 'invalid_masks.json'), 'w') as f:
        json.dump(sorted(list(invalid_masks_set)), f, indent=4)
    invalid_images_set = set([x.strip('.jpg') for x in invalid_images_set])
    invalid_masks_set = set([x.strip('.png') for x in invalid_masks_set])
    invalid_set = invalid_images_set & invalid_masks_set
    with open(os.path.join(args.data_path, 'invalid_set.json'), 'w') as f:
        json.dump(sorted(list(invalid_set)), f, indent=4)

if __name__ == "__main__":
    main()