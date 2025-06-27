import numpy as np
import torch
from network.styleunet.dual_styleunet import StyleUNet


if __name__ == "__main__":
    inp_size = 512
    out_size = 1024
    net = StyleUNet(inp_size = inp_size, inp_ch = 3, out_ch = 20, out_size = out_size, style_dim = 512, n_mlp = 2, middle_size=64)
    net.to('cuda:0')
    style = torch.ones([1, net.style_dim], dtype=torch.float32, device='cuda:0') / np.sqrt(net.style_dim)
    inp = torch.randn(1, 3, inp_size, inp_size, device='cuda:0')
    view_feature = torch.randn(1, 128, 128, 128, device='cuda:0')
    out, _ = net([style], inp, randomize_noise = False, view_feature = view_feature)
    print(out.shape)