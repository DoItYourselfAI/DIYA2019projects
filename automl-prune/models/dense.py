import torch
import torch.nn as nn

from modules.block import BasicBlock

__all__ = ['dense3', 'dense11']


class DenseMLP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        config = {
            'input_dim': None,
            'output_dim': None,
            'hidden_dim': None,
            'num_layers': 3,
            'init': True
        }
        for k, v in config.items():
            setattr(self, k, kwargs.get(k, v))
        assert self.num_layers >= 2

        self.layers = self.create_layers()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        ctx = torch.zeros(
            (self.num_layers, x.size(0), self.hidden_dim),
            dtype=x.dtype,
            device=x.get_device()
        )
        for idx_from in range(self.num_layers - 1):
            x = ctx[idx_from].clone() if idx_from != 0 else x
            for idx_to in range(idx_from + 1, self.num_layers):
                ctx[idx_to] = ctx[idx_to].clone() + self.layers[idx_from][idx_to](x)
        out = self.layers[-1](ctx[-1])
        return out
        
    def create_layers(self):
        layers = nn.ModuleList()
        for idx_from in range(self.num_layers - 1):
            blocks = nn.ModuleList()
            for idx_to in range(self.num_layers):
                if idx_to > idx_from:
                    if idx_from == 0:
                        block = BasicBlock(
                            in_features=self.input_dim,
                            out_features=self.hidden_dim,
                            init=self.init
                        )
                    else:
                        block = BasicBlock(
                            in_features=self.hidden_dim,
                            out_features=self.hidden_dim,
                            init=self.init
                        )
                    blocks.append(block)
                else:
                    blocks.append(None)
            layers.append(blocks)
        
        last_layer = nn.Linear(self.hidden_dim, self.output_dim)
        if self.init:
            nn.init.xavier_normal_(last_layer.weight)
        layers.append(last_layer)
        return layers


def dense3(args):
    if args.env == 'mnist':
        input_dim = 1*28*28
        output_dim = 10
    elif args.env == 'imdb':
        input_dim = 30*50
        output_dim = 2
    else:
        raise NotImplementedError

    config = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_dim': 2*input_dim,
        'num_layers': 3,
        'init': args.init
    }
    return DenseMLP(**config)

def dense11(args):
    if args.env == 'mnist':
        input_dim = 1*28*28
        output_dim = 10
    elif args.env == 'imdb':
        input_dim = 30*50
        output_dim = 2
    else:
        raise NotImplementedError

    config = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_dim': 2*input_dim,
        'num_layers': 11,
        'init': args.init
    }
    return DenseMLP(**config)