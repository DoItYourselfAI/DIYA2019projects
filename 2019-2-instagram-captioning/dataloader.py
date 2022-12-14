

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from config import Config


target_size = 224
default_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])


class DGUDataset(Dataset):
    """
    Custom dataset designed to load DGU Dataset using either one of ['total.json', 'train.json', 'val.json', 'test.json']
    """
    def __init__(self, json_file, vocab, transform=default_transform, type='hashtag', tokenize_fn=None,
                 load_on_ram=False):
        """
        json_file: path to json file
        vocab: dictionary object. must include <PAD>, <UNK>, <start>, <end> as its keys.
        transform: transform to be applied to image
        type: 'hashtag' or 'text'
        tokenize_fn: function that is used to tokenize text when type is 'text'. input: string, output: list of strings.
        """
        
        assert any(json_file.endswith(file_type) for file_type in ['total.json', 'train.json', 'val.json', 'test.json'])
        assert type in ['hashtag', 'text']
        assert '<UNK>' in vocab
        assert '<start>' in vocab
        assert '<end>' in vocab
        
        if type == 'text':
            assert tokenize_fn is not None
        self.vocab = vocab
            
        self.root_dir = '/'.join(json_file.split('/')[:-1])
        self.type = type
        self.vocab = vocab
        
        self.transform = transform
        self.tokenize_fn = tokenize_fn
        self.load_on_ram = load_on_ram
        
        with open(json_file) as fr:
            # remove empty data
            d = json.load(fr)
            self.json = []
            for item in d:
                if type not in item:
                    continue
                if type == 'text':
                    # vocab??? ???????????? ????????? ????????? ??????????????? ???????????? ??????
                    target = [token for token in tokenize_fn(item['text']) if token in self.vocab]
                else:
                    target = [t for t in item['hashtag'] if t in self.vocab]
                if target != []:
                    # ????????? ?????? ??? ?????? ??????...
                    if type == 'text':
                        if len(target) > Config.max_text_length:
                            continue
                    elif type == 'hashtag':
                        if len(target) > Config.max_hashtag_length:
                            continue
                    self.json.append(item)
                    
        # target ?????? ??????
        self.targets = []
        self.tokens = []
        for item in self.json:
            if type == 'text':
                text = item['text']
                tokens = self.tokenize_fn(text)
                self.tokens.append(tokens)
                UNK_idx = self.vocab['<UNK>']
                target = [self.vocab.get(token, UNK_idx) for token in tokens]
            elif type == 'hashtag':
                # ??????????????? ??????, vocab??? ???????????? ????????? ?????? ??????
                hashtags = item['hashtag']
                self.tokens.append(hashtags)
                target = [self.vocab.get(hashtag) for hashtag in hashtags if hashtag in self.vocab]
            self.targets.append(target)
        
        # ???????????? ?????? ????????????
        if self.load_on_ram:
            self.images = []
            for item in self.json:
                image = Image.open(os.path.join(self.root_dir, item['image_path']))
                if self.transform is not None:
                    image = self.transform(image)
                self.images.append(image)
                    
    def __getitem__(self, index):
        item = self.json[index]
        
        # load target = hashtag or text
        target = self.targets[index]
        target = [self.vocab.get('<start>')] + target + [self.vocab.get('<end>')]
        target = torch.LongTensor(target)
        
        # load image
        if self.load_on_ram:
            image = self.images[index]
        else:
            image = Image.open(os.path.join(self.root_dir, item['image_path']))
            if self.transform is not None:
                image = self.transform(image)
            
        return image, target
    
    def __len__(self):
        return len(self.json)
    
    
def collate_fn(data):
    """
    data: list of tuple (image, target)
    
    returns:
        images: torch tensor of shape (batch_size, 3, img_height, img_width)
        targets: torch tensor of shape (batch_size, padded_length)
        lengths: list of 'real' length of each target
    """
    
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, targets = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge targets (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in targets]
    padded_targets = torch.zeros(len(targets), max(lengths)).long()
    for i, cap in enumerate(targets):
        end = lengths[i]
        padded_targets[i, :end] = cap[:end]        
    return images, padded_targets, lengths


def get_dataloader(json_file, vocab, transform=default_transform, type='hashtag', tokenize_fn=None,
                   batch_size=32, shuffle=True, num_workers=-1, load_on_ram=False):
    
    dataset = DGUDataset(json_file, vocab, transform=transform, type=type, tokenize_fn=tokenize_fn, load_on_ram=load_on_ram)
    
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    
    return loader
        