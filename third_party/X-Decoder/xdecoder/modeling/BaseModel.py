import os
import logging

import torch
import torch.nn as nn

from xdecoder.utils.model import align_and_update_state_dicts

logger = logging.getLogger(__name__)


def _checkpoint_map_location(opt):
    """Load tensors on CPU when CUDA is unavailable (avoids torch.load cuda deserialize errors)."""
    loc = opt.get("device", "cpu")
    loc_str = str(loc)
    if loc_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return loc


class BaseModel(nn.Module):
    def __init__(self, opt, module: nn.Module):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.model = module

    def forward(self, *inputs, **kwargs):
        outputs = self.model(*inputs, **kwargs)
        return outputs

    def save_pretrained(self, save_dir):
        torch.save(self.model.state_dict(), os.path.join(save_dir, "model_state_dict.pt"))

    def from_pretrained(self, load_dir):
        state_dict = torch.load(load_dir, map_location=_checkpoint_map_location(self.opt))
        state_dict = align_and_update_state_dicts(self.model.state_dict(), state_dict)
        self.model.load_state_dict(state_dict, strict=False)
        return self

    def from_pretrained_seg(self, load_dir):
        state_dict = torch.load(load_dir, map_location=_checkpoint_map_location(self.opt))
        state_dict = align_and_update_state_dicts(self.model.sem_seg_head.state_dict(), state_dict)
        self.model.sem_seg_head.load_state_dict(state_dict, strict=False)
        return self