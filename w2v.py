import torch as t
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import numpy as np
from collections import Counter
import copy
import chainer


class Net(nn.Module):
    def __init__(self, vocab_num, embed_num, batch_size=1):
        super(Net, self).__init__()
        self.W_center = nn.Embedding(vocab_num, embed_num)
        self.W_context = nn.Embedding(vocab_num, embed_num)

        init_range = 0.5 / embed_num

        self.W_center.weight.data.uniform_(-init_range, init_range)
        self.W_context.weight.data.uniform_(-0.0001, 0.0001)

        self.batch_size = batch_size

    def forward(self, center, context, negative):
        embed_center = self.W_center(center)
        embed_context = self.W_context(context)
        # print(context)
        embed_negative = self.W_context(negative)
        # print(embed_context, embed_center)

        # embed_centerは、(batch_size,1)
        # embed_contextは、(batch_size,context_size,embed_size)
        # embed_negativeは、(batch_size,negative_size,embed_size)

        pos_outs = F.logsigmoid(
            t.sum(t.bmm(embed_context,
                        embed_center.unsqueeze(2)), dim=1)).squeeze()
        neg_outs = F.logsigmoid(
            t.sum(-t.bmm(embed_negative,
                         embed_center.unsqueeze(2)), dim=1)).squeeze()

        loss = pos_outs + neg_outs

        # 最大化するのでマイナス(ほんとはcontextとnegativeの大きさでも割るべき)
        return -loss.sum() / self.batch_size


# ネガティブサンプリング用のやつ
# corpusは
# [[{単語のid},...],[..]...]
# target以外の単語をランダムに抽出


def sampler(corpus, context, sample_size):
    all_word = [w for l in corpus for w in l if w not in context]
    len_all_word = len(all_word)
    dic = np.array([w for w, c in Counter(all_word).most_common()])
    rate_list = [c / len_all_word for _, c in
                 Counter(all_word).most_common()]
    # random.choiceを使うと遅いらしい
    return dic[np.random.choice(len(rate_list), p=rate_list,
                                size=sample_size, replace=False)]


def get_dict(corpus):
    return {w: i for i, w in enumerate(
        set([w for l in [c.split(" ") for c in corpus] for w in l]))}


def get_data(c_id):
    # corpus_id = [[dic[w] for w in c.split(" ")] for c in corpus]
    # print(corpus_id.shape)
    data = []
    # for i, c_id in enumerate(corpus_id):
    len_c_id = c_id.shape[0] - 1
    for j, w_id in enumerate(c_id):
        pos = []
        if j > 0:
            pos.append(c_id[j - 1])
        if j < len_c_id:
            pos.append(c_id[j + 1])
        neg = sampler([c_id], pos, 2)
        data.append([w_id, np.array(pos), neg])
    return data


if __name__ == "__main__":

    corpus = [
        "he is a king",
        "she is a queen",
        "he is a man",
        "she is a woman",
        "warsaw is poland capital",
        "berlin is germany capital",
        "paris is france capital"
    ]

    train, val, test = chainer.datasets.get_ptb_words()
    # dic = chainer.datasets.get_ptb_words_vocabulary()
    # dic = get_dict(corpus)
    data = get_data(train)
    trainloader = t.utils.data.DataLoader(
        data, batch_size=1, shuffle=True)

    net = Net(10000, 50)
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    for epoch in range(30):
        loss = 0
        for c, p, n in trainloader:
            optimizer.zero_grad()
            loss = net(c, p, n)
            loss.backward()
            optimizer.step()
            loss += loss.item()
        print(epoch, loss)
