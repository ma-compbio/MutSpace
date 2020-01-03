import torch
from torch import nn
from torch.optim import Adam


class InnerProductSimilarity(nn.Module):
    def __init__(self, temp):
        super(InnerProductSimilarity, self).__init__()
        self.temp = temp

    def forward(self, a, b):
        d = a.shape[1]
        a = a.unsqueeze(1)  # N x 1 x dim
        if len(b.shape) == 2:
            b = b.unsqueeze(2)  # N x dim x 1
            similarity = torch.bmm(a, b).squeeze()
        elif len(b.shape) == 3:
            # N x neg x dim
            similarity = torch.sum(a * b, dim=(-1,))
        else:
            assert False
        return similarity / pow(d, self.temp)   # [N] or [N x n_neg]


class MarginRankingLoss(nn.Module):
    def __init__(self, margin=1., aggregate=torch.mean):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.aggregate = aggregate

    def forward(self, positive_similarity, negative_similarity, negative_mask):
        """
        :param positive_similarity: [N]
        :param negative_similarity: [N x K]
        :param negative_mask: [N x K]
        :return:
        """
        positive_similarity = positive_similarity.unsqueeze(1)
        return self.aggregate(
            torch.clamp((self.margin - positive_similarity + negative_similarity) * negative_mask, min=0))


class MutSpace(nn.Module):
    def __init__(self, config, n_features):
        super(MutSpace, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(n_features, config.emb_dim, max_norm=config.max_norm)
        self.similarity = InnerProductSimilarity(config.temp)
        self.loss = MarginRankingLoss(margin=config.margin)
        self.optimizer = Adam(self.parameters(), lr=config.lr)

    def forward(self, batch):
        neg_mask = batch['neg_mask']            # [N x n_neg]
        pos_a = self.embedding(batch['pos_a'])  # [N x n_a x d]
        pos_b = self.embedding(batch['pos_b'])  # [N x n_b x d]
        neg_b = self.embedding(batch['neg_b'])  # [N x n_neg x n_b x d]
        pos_a = pos_a.sum(dim=1) / pow(pos_a.shape[1], 0.5)
        pos_b = pos_b.sum(dim=1) / pow(pos_b.shape[1], 0.5)
        neg_b = neg_b.sum(dim=2) / pow(neg_b.shape[2], 0.5)
        pos_score = self.similarity(pos_a, pos_b)
        neg_score = self.similarity(pos_a, neg_b)
        return self.loss(pos_score, neg_score, neg_mask), pos_score, neg_score

    def train_batch(self, batch):
        self.train()
        loss, pos_score, neg_score = self.forward(batch)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), pos_score.mean().item(), neg_score.mean().item()
