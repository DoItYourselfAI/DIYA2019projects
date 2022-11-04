from abc import ABC, abstractmethod
from utils.logger import Logger


class Agent:
    def __init__(self, name, model, env, args=None, **kwargs):
        config = {
            'name': name,
            'model': model,
            'env': env,
            'logger': None,
            'lvl': 21,
            'color': 'white',
            'args': args
        }
        for k, v in config.items():
            setattr(self, k, kwargs.get(k, v))

        set_level = self.logger is not None
        while set_level:
            try:
                self.logger.add_level(name, self.lvl, self.color)
                set_level = False
            except:
                self.lvl += 1
                
        self.step = 0
        self.epoch = 0

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def infer(self):
        pass