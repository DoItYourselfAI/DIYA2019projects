import os
import json
import random
import numpy as np
import torch

from utils.common import *
from utils.logger import Logger

import agents
import envs
import models


def train(args):
    logger = Logger(args)
    model = load_model(args)
    env = getattr(envs, args.env)(args)
    agent = getattr(agents, args.agent)(model, env, args, logger=logger)

    logger.log("Validation before training")
    info = agent.infer()
    best_loss = info.avg['Loss']
    for epoch in range(1, args.num_epochs + 1):
        logger.log("Begin training for epoch: {}".format(epoch))
        info = agent.train()

        logger.log("Validation for epoch: {}".format(epoch))
        info = agent.infer()
        loss = info.avg['Loss']
        if loss < best_loss:
            best_loss = loss
            logger.log("Saving best model...")
            save_model(agent.model, args)


def test(args):
    pass


if __name__ == "__main__":
    with open('config.json') as config:
        args = json.load(config)
    args = ArgumentParser(args)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    args.init = args.checkpoint is None
    if args.mode == 'train':
        train(args)
    else:
        test(args)