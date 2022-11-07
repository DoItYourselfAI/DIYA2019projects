import os
import pickle
import time
import json
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from settings import *
from utils.common import ArgumentParser
from utils.logger import Logger
from utils.summary import EvaluationMetrics


def train(args):
    from loaders import cooccur_loader
    from model import RecSys

    logger = Logger('TRAIN', verbose=True, args=args)

    model = RecSys(args)
    if args.checkpoint is not None:
        logger.log("loading model from {}...".format(args.checkpoint))
        path = os.path.join(PROJECT_ROOT, PTH_DIR, args.checkpoint)
        model.load_state_dict(torch.load(path))
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        nesterov=True,
        weight_decay=args.weight_decay
    )

    info = EvaluationMetrics(['Epoch', 'Time/Step', 'Time/Item', 'Loss'])
    loader = cooccur_loader('train', args)
    step = 0
    for epoch in range(args.num_epochs):
        for user, user_idx, item, item_idx, read in loader:
            step += 1
            st = time.time()

            optimizer.zero_grad()
            loss = model.compute_loss(user, item, read)
            loss.backward()
            optimizer.step()

            elapsed = time.time() - st
            info.update('Epoch', epoch)
            info.update('Time/Step', elapsed)
            info.update('Time/Item', elapsed/read.size(0))
            info.update('Loss', loss.item())

            if step % args.log_step == 0:
                logger.scalar_summary(info.avg, step)
                info.reset()
        
        path = os.path.join(PROJECT_ROOT, PTH_DIR)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'model.{}.pth'.format(epoch+1))
        torch.save(model.state_dict(), path)


def dev(args):
    from loaders import cooccur_loader
    from model import RecSys
    
    logger = Logger('DEV', verbose=True, args=args)
    
    model = RecSys(args)
    if args.checkpoint is not None:
        logger.log("loading model from {}...".format(args.checkpoint))
        path = os.path.join(PROJECT_ROOT, PTH_DIR, args.checkpoint)
        model.load_state_dict(torch.load(path))
    model.eval()

    loader = cooccur_loader('dev', args)
    scores = {}
    for epoch in range(args.num_epochs):
        for user, user_idx, item, item_idx, read in loader:
            score = model(user, item).detach().cpu().numpy()
            for i in range(len(score)):
                new_array = np.array([item_idx[i].cpu().numpy(), score[i]])
                if user_idx[i].cpu().numpy() not in scores:
                    scores[user_idx[i].cpu().numpy()] = new_array
                else:
                    scores[user_idx[i]] = np.stack(scores[user_idx[i]], new_array)
            print(scores)
            raise




    item_path = os.path.join(PROJECT_ROOT, RES_DIR, 'metadata.json')
    item_ids = pd.read_json(item_path, lines=True)["id"]

    read_path = os.path.join(PROJECT_ROOT, TMP_DIR, 'read.pkl')
    with open(read_path, 'rb') as f:
        read = pickle.load(f)

    items = item_loader(args)
    users = UserIter('dev', args)

    result_path = os.path.join(PROJECT_ROOT, TMP_DIR, 'recommend.txt')
    with open(result_path, 'w') as f:
        for user_idx, user in enumerate(users):
            line = [users.user_list[user_idx]]
            scores = []
            for item_idx, item in enumerate(logger.progress(items)):
                user = (torch.cat([user[0].unsqueeze(0)]*item[0].size(0)),
                        torch.cat([user[1].unsqueeze(0)]*item[0].size(0)))
                score = list(model(user, item).detach().cpu().numpy())
                scores.extend(score)
            
            for idx in range(len(scores)):
                if user_idx in read and idx in read[user_idx]:
                    scores[idx] = -np.inf
            recs = np.argsort(-np.asarray(scores))[:100]
            for rec in recs:
                line.append(item_ids[rec])
            print(line)
            raise


    
    scores = np.stack(scores)



if __name__ == "__main__":
    with open('config.json') as config:
        args = json.load(config)
    args = ArgumentParser(args)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)    

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.mode == 'train':
        train(args)
    elif args.mode == 'dev':
        dev(args)
    elif args.mode == 'test':
        test(args)
