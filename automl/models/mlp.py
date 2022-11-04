import torch
import torch.nn as nn

from modules.block import BasicBlock

__all__ = ['mlp3']


class MLP(nn.Module):
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
        for layer in self.layers:
            x = layer(x)
        return x
        
    def create_layers(self):
        layers = nn.ModuleList()
        layers.append(
            BasicBlock(
                in_features=self.input_dim,
                out_features=self.hidden_dim,
                init=self.init
            )
        )
        for idx in range(1, self.num_layers - 1):
            layers.append(
                BasicBlock(
                    in_features=self.hidden_dim,
                    out_features=self.hidden_dim,
                    init=self.init
                )
            )
        last_layer = nn.Linear(self.hidden_dim, self.output_dim)
        if self.init:
            nn.init.xavier_normal_(last_layer.weight)
        layers.append(last_layer)
        return layers


def mlp3(args):
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
    return MLP(**config)