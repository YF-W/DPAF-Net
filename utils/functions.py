import torch
import numpy as np
import random
import logging

logger = logging.getLogger('MSA')

def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key])
    return dst_str

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def assign_gpu(gpu_ids, memory_limit=1e16):
    index = int(gpu_ids[0]) if gpu_ids else 0
    return torch.device(f'cuda:{index}' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    res = 0
    for p in model.parameters():
        if p.requires_grad:
            res += p.numel()

    return res
