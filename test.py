import torch
import numpy as np
import argparse
import os
import torch.backends.cudnn as cudnn
from architecture.model import generator
from dfc2018_dataset import TestDataset
from torch.utils.data import DataLoader
from osgeo import gdal
import warnings

def warn(*args, **kwargs):
    pass
warnings.warn = warn

def write_img(filename, im_data):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--data_root', type=str, default='/mnt/dataset/liuxinyang/DFC2018')
parser.add_argument('--pretrained_model_path', type=str, default='')
parser.add_argument('--outf', type=str, default='/mnt/dataset/liuxinyang/SSTAT-main/test_results')
parser.add_argument("--gpu_id", type=str, default='0')
parser.add_argument('--test_flag', default='test')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

# load dataset
test_data = TestDataset(data_root=opt.data_root, bgr2rgb=True)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


save_root = opt.outf
os.makedirs(save_root, exist_ok=True)
save_dir = os.path.join(save_root, opt.test_flag)
os.makedirs(save_dir, exist_ok=True)

def test(test_loader, model):
    model.eval()

    for i, (input, target) in enumerate(test_loader):
        input = input.cuda()
        with torch.no_grad():
            output = model(input)
            output_denorm = output * 4095

        # output: [1,C,H,W] -> [C,H,W]
        out_np = output_denorm[0].detach().cpu().numpy()  # float, [C,H,W]
        tif_data = np.clip(out_np, 0, 65535).round().astype(np.uint16)

        base_name = os.path.basename(test_data.multi_path[i])  # e.g., "0001.tif"
        save_path = os.path.join(save_dir, base_name)
        write_img(save_path, tif_data)

        if (i + 1) % 10 == 0 or (i + 1) == len(test_loader):
            print(f"[{i + 1}/{len(test_loader)}] saved: {save_path}")

    return

if __name__ == '__main__':
    cudnn.benchmark = True
    pretrained_model_path = opt.pretrained_model_path
    model = generator(input_nc=3, embed_dim=256, output_nc=48).cuda()
    checkpoint = torch.load(pretrained_model_path, map_location='cuda')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    # torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, 'module.')
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    test(test_loader, model)

