import os
import torch

import models
from settings import PROJECT_ROOT, PTH_DIR

__all__ = ['ArgumentParser']
__all__ += ['save_model', 'load_model']


class ArgumentParser(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def save_model(model, args):
    path = os.path.join(PROJECT_ROOT, PTH_DIR, args.tag)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    path = os.path.join(path, args.model + '.pth')
    info = {
        'model': model.state_dict(),
        'args': args
    }
    torch.save(info, path)


def load_model(args):
    model = getattr(models, args.model)(args)
    if args.checkpoint is not None:
        path = os.path.join(PROJECT_ROOT, PTH_DIR, args.checkpoint + '.pth')
        ckpt = torch.load(path, map_location=lambda storage, loc: storage)
        state_dict = ckpt['model']
        model.load_state_dict(state_dict)
    
    if args.half:
        model = model.half()
    model = model.to(args.device)
    return model