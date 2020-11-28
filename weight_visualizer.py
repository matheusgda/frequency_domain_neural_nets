import sys

import torch
import torch.fft as fft

import utils
import fdnn

argc = len(sys.argv)
state_dict_path = sys.argv[1]
f_index = int(sys.argv[2])

f = open(state_dict_path, 'rb')
state_dict = torch.load(f, map_location=torch.device('cpu'))
f.close()
keys = list(state_dict.keys())

keyr = 'preserving_block.sequential.11.freal2'
keyi = 'preserving_block.sequential.11.fimag2'

wr = state_dict[keyr][:, :, :, f_index]
wi = state_dict[keyi][:, :, :, f_index]
w = torch.stack((wr, wi), dim=-1)
# print(w)
print(wr.size())
print(wi.size())
print(w.size())

process = lambda x : fft.ifftn(torch.view_as_complex(x), dim=(0, 1))
utils.show_conv_weight(w, process)
