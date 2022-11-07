import torch
import torch.nn as nn


class RecSys(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.alpha = args.alpha
        self.keyword_embedding = nn.Linear(372226, args.embedding_size)
        self.author_embedding = nn.Linear(19065, args.embedding_size) 
        self.magazine_embedding = nn.Linear(28028, args.embedding_size)
        self.tag_embedding = nn.Linear(86037, args.embedding_size)

        self.user_network = nn.Sequential(
            nn.BatchNorm1d(2*args.embedding_size),
            nn.Linear(2*args.embedding_size, args.embedding_size),
            nn.ReLU(),
            nn.Linear(args.embedding_size, args.embedding_size)
        )
        self.item_network = nn.Sequential(
            nn.BatchNorm1d(3*args.embedding_size),
            nn.Linear(3*args.embedding_size, args.embedding_size),
            nn.ReLU(),
            nn.Linear(args.embedding_size, args.embedding_size)
        )
        self.to(args.device)
        self.args = args

    def forward(self, user, item):
        keywords, following = user
        keywords = keywords.to(self.args.device)
        following = following.to(self.args.device)

        user_embedding = torch.cat([
            self.keyword_embedding(keywords),
            self.author_embedding(following)],
            dim=-1
        )
        user_embedding = self.user_network(user_embedding)

        author, magazine, tag = item
        author = author.to(self.args.device)
        magazine = magazine.to(self.args.device)
        tag = tag.to(self.args.device)

        item_embedding = torch.cat([
            self.author_embedding(author),
            self.magazine_embedding(magazine),
            self.tag_embedding(tag)],
            dim=-1
        )
        item_embedding = self.item_network(item_embedding)
        
        return torch.sum(user_embedding*item_embedding, dim=-1)

    def compute_loss(self, user, item, read):
        score = self.forward(user, item)
        read = read.to(self.args.device).float()
        mse = torch.norm((score - read), dim=-1)
        adjusted_mse = torch.sum(mse*read)*self.alpha + torch.sum(mse*(1 - read))
        adjusted_mse = adjusted_mse/read.size(0)
        return adjusted_mse
