import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def criterion(model, labels, outs, A, device=None, args=None):

    ce_loss = F.cross_entropy(outs, labels)

    # 论文中A为(r, n)，计算A × AT
    # 这里的A为(n, r)，因此应该计算AT × A
    penalization_loss = torch.norm(torch.bmm(torch.permute(A, (0, 2, 1)), A) - torch.eye(args.aspects).to(device))
    
    return ce_loss, penalization_loss