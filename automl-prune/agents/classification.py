import time
import numpy as np
import torch
import torch.nn as nn

from utils.logger import Logger
from utils.summary import EvaluationMetrics
from agents.base import Agent

__all__ = ['cross_entropy']


class CrossEntropy(Agent):
    def __init__(self, name, model, env, args, **kwargs):
        super().__init__(name, model, env, args, **kwargs)
        config = {
            'optimizer': None,
            'scheduler': None,
            'l1_decay': 0,
            'l2_decay': 1e-5
        }
        for k, v in config.items():
            setattr(self, k, kwargs.get(k, v))
        
        self.train_loader = env['train']
        self.val_loader = env['val']
        
        self.criterion = nn.CrossEntropyLoss()
        self.info = EvaluationMetrics(
            [
                'Time/Step',
                'Time/Item',
                'Loss',
                'Top1',
            ]
        )
        
    def train(self):
        for idx, (data, label) in enumerate(self.train_loader):
            st = time.time()
            self.step += 1
            self.model.train()

            data = data.to(self.args.device)
            if self.args.half:
                data = data.half()
            label = label.to(self.args.device)
    
            output = self.model(data)

            params = filter(lambda p: p.requires_grad, self.model.parameters())
            params = torch.cat([x.view(-1) for x in params])
            ce_loss = self.criterion(output, label)
            l1_loss = self.l1_decay*torch.norm(params, 1)
            l2_loss = self.l2_decay*torch.norm(params, 2)
            loss = ce_loss + l1_loss + l2_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            elapsed = time.time() - st
            self.info.update('Time/Step', elapsed)
            self.info.update('Time/Item', elapsed/data.size(0))
            self.info.update('Loss', ce_loss.item())

            _, pred = torch.max(output, -1)
            top1 = (label == pred).float().mean().item()
            self.info.update('Top1', top1)

            if self.step % self.args.log_step == 0:
                self.logger.scalar_summary(self.info.avg, self.step, self.name)
                self.info.reset()

        self.epoch += 1
        self.scheduler.step()
    
        return self.info

    def infer(self):
        info = EvaluationMetrics(
            [
                'Time/Step',
                'Time/Item',
                'Loss',
                'Top1',
            ]
        )
        for idx, (data, label) in enumerate(self.val_loader):
            st = time.time()
            self.model.eval()

            data = data.to(self.args.device)
            if self.args.half:
                data = data.half()
            label = label.to(self.args.device)
                
            output = self.model(data)

            params = filter(lambda p: p.requires_grad, self.model.parameters())
            params = torch.cat([x.view(-1) for x in params])
            ce_loss = self.criterion(output, label)
            l1_loss = self.l1_decay*torch.norm(params, 1)
            l2_loss = self.l2_decay*torch.norm(params, 2)
            loss = ce_loss + l1_loss + l2_loss

            elapsed = time.time() - st
            info.update('Time/Step', elapsed)
            info.update('Time/Item', elapsed/data.size(0))
            info.update('Loss', ce_loss.item())

            _, pred = torch.max(output, -1)
            top1 = (label == pred).float().mean().item()
            info.update('Top1', top1)

        self.logger.scalar_summary(info.avg, self.step, self.name)

        return info


def cross_entropy(model, env, args=None, **kwargs):
    params = filter(lambda p: p.requires_grad, model.parameters())
    kwargs['optimizer'] = torch.optim.SGD(
        params,
        lr=args.learning_rate,
        momentum=args.momentum,
        nesterov=True
    )
    kwargs['scheduler'] = torch.optim.lr_scheduler.MultiStepLR(
        kwargs['optimizer'],
        [int(args.num_epochs*0.5), int(args.num_epochs*0.75)],
        gamma=0.1
    )
    kwargs['l1_decay'] = args.weight_decay
    kwargs['l2_decay'] = 0
    return CrossEntropy('CE', model, env, args, **kwargs)