
from model.D3DPoseMamba import D3DPoseMamba_dffvt, D3DPoseMamba
from model.PoseMamba import PoseMamba
from model.SasMamba import SasPoseMamba
from utils.activation import SymSum
from torch import nn
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





def load_pretrained_weights(model, checkpoint):

    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict:
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=True)
    print(f'[INFO] (load_pretrained_weights) {len(matched_layers)} layers are loaded')
    print(f'[INFO] (load_pretrained_weights) {len(discarded_layers)} layers are discared')


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def decay_lr_exponentially(lr, lr_decay, optimizer):
    lr *= lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return lr




def load_model_ssdmamba(args):
    act_mapper = {
        "gelu": nn.GELU,
        'relu': nn.ReLU,
        'symsum': SymSum
    }
    if args.model_name == "SSDMamba_vt":
        model = D3DPoseMamba_dffvt(
            num_frame=args.n_frames,
            embed_dim_ratio=args.dim_feat,
            mlp_ratio=args.mlp_ratio,
            depth=args.n_layers,
            freq_l=9
            )
    elif args.model_name == "SSDMamba":
        model = D3DPoseMamba(
            num_frame=args.n_frames,
            embed_dim_ratio=args.dim_feat,
            mlp_ratio=args.mlp_ratio,
            depth=args.n_layers,
            num_joints=args.num_joints
        )
    elif args.model_name == "PoseMamba":
        model = PoseMamba(
            num_frame=args.n_frames,
            embed_dim_ratio=args.dim_feat,
            mlp_ratio=args.mlp_ratio,
            depth=args.n_layers,
            num_joints=args.num_joints,
            in_chans=args.in_chans,
            ssm_conv=args.ssm_conv,
        )
    elif args.model_name == "SasMamba":
        model = SasPoseMamba(
            num_frame=args.n_frames,
            embed_dim_ratio=args.dim_feat,
            mlp_ratio=args.mlp_ratio,
            depth=args.n_layers,
            num_joints=args.num_joints,
            in_chans=args.in_chans,
            ssm_conv=args.ssm_conv,
        )
    else:
        raise Exception("Undefined model name")

    return model


