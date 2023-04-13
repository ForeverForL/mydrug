import numpy as np
import torch
from torch import nn
from torch import optim
import tqdm
import utils

class RNNGen(nn.Module):
    """
    基础RNN模型
    """
    def __init__(self, voc, embed_size=128, hidden_size=512, is_lstm=True, lr=1e-3):
        super(RNNGen, self).__init__()
        self.voc = voc
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = voc.size

        self.embed = nn.Embedding(voc.size, embed_size)
        self.is_lstm = is_lstm
        rnn_layer = nn.LSTM if is_lstm else nn.GRU
        self.rnn = rnn_layer(embed_size, hidden_size, num_layers=3, batch_first=True)
        self.linear = nn.Linear(hidden_size, voc.size)
        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.to(utils.dev)

    def forward(self, input, h):
        output = self.embed(input.unsqueeze(-1))
        output, h_out = self.rnn(output, h)
        output = self.linear(output).squeeze(1)
        return output, h_out

    def init_h(self, batch_size, labels=None):
        h = torch.rand(3, batch_size, 512).to(utils.dev)
        if labels is not None:
            h[0, batch_size, 0] = labels
        if self.is_lstm:
            c = torch.rand(3, batch_size, self.hidden_size).to(utils.dev)
        return (h, c) if self.is_lstm else h

    def likelihood(self, target):
        batch_size, seq_len = target.size()
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(utils.dev)
        h = self.init_h(batch_size)
        scores = torch.zeros(batch_size, seq_len).to(utils.dev)
        for step in range(seq_len):
            logits, h = self(x, h)
            logits = logits.log_softmax(dim=-1)
            score = logits.gather(1, target[:, step:step+1]).squeeze()
            scores[:, step] = score
            x = target[:, step]
        return scores

    def PGLoss(self, loader):
        for seq, reward in loader:
            self.zero_grad()
            score = self.likelihood(seq)
            loss = score * reward
            loss = -loss.mean()
            loss.backward()
            self.optim.step()

    def sample(self, batch_size):
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(utils.dev)
        h = self.init_h(batch_size)
        sequences = torch.zeros(batch_size, self.voc.max_len).long().to(utils.dev)
        isEnd = torch.zeros(batch_size).bool().to(utils.dev)

        for step in range(self.voc.max_len):
            logit, h = self(x, h)
            proba = logit.softmax(dim=-1)
            x = torch.multinomial(proba, 1).view(-1)
            x[isEnd] = self.voc.tk2ix['EOS']
            sequences[:, step] = x

            end_token = (x == self.voc.tk2ix['EOS'])
            isEnd = torch.ge(isEnd + end_token, 1)
            if (isEnd == 1).all(): break
        return sequences

    def evolve(self, batch_size, epsilon=0.01, crover=None, mutate=None):
        # Start tokens
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(utils.dev)
        # Hidden states initialization for exploitation network
        h = self.init_h(batch_size)
        # Hidden states initialization for exploration network
        h1 = self.init_h(batch_size)
        h2 = self.init_h(batch_size)
        # Initialization of output matrix
        sequences = torch.zeros(batch_size, self.voc.max_len).long().to(utils.dev)
        # labels to judge and record which sample is ended
        is_end = torch.zeros(batch_size).bool().to(utils.dev)

        for step in range(self.voc.max_len):
            logit, h = self(x, h)
            proba = logit.softmax(dim=-1)
            if crover is not None:
                ratio = torch.rand(batch_size, 1).to(utils.dev)
                logit1, h1 = crover(x, h1)
                proba = proba * ratio + logit1.softmax(dim=-1) * (1 - ratio)
            if mutate is not None:
                logit2, h2 = mutate(x, h2)
                is_mutate = (torch.rand(batch_size) < epsilon).to(utils.dev)
                proba[is_mutate, :] = logit2.softmax(dim=-1)[is_mutate, :]
            # sampling based on output probability distribution
            x = torch.multinomial(proba, 1).view(-1)

            is_end |= x == self.voc.tk2ix['EOS']
            x[is_end] = self.voc.tk2ix['EOS']
            sequences[:, step] = x
            if is_end.all(): break
        return sequences

    def evolve1(self, batch_size, epsilon=0.01, crover=None, mutate=None):
        # Start tokens
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(utils.dev)
        # Hidden states initialization for exploitation network
        h = self.init_h(batch_size)
        # Hidden states initialization for exploration network
        h2 = self.init_h(batch_size)
        # Initialization of output matrix
        sequences = torch.zeros(batch_size, self.voc.max_len).long().to(utils.dev)
        # labels to judge and record which sample is ended
        is_end = torch.zeros(batch_size).bool().to(utils.dev)

        for step in range(self.voc.max_len):
            is_change = torch.rand(1) < 0.5
            if crover is not None and is_change:
                logit, h = crover(x, h)
            else:
                logit, h = self(x, h)
            proba = logit.softmax(dim=-1)
            if mutate is not None:
                logit2, h2 = mutate(x, h2)
                ratio = torch.rand(batch_size, 1).to(utils.dev) * epsilon
                proba = logit.softmax(dim=-1) * (1 - ratio) + logit2.softmax(dim=-1) * ratio
            # sampling based on output probability distribution
            x = torch.multinomial(proba, 1).view(-1)

            x[is_end] = self.voc.tk2ix['EOS']
            sequences[:, step] = x

            # Judging whether samples are end or not.
            end_token = (x == self.voc.tk2ix['EOS'])
            is_end = torch.ge(is_end + end_token, 1)
            #  If all of the samples generation being end, stop the sampling process
            if (is_end == 1).all(): break
        return sequences

    def fit(self, loader_train, out, loader_valid=None, epochs=100, lr=1e-3):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        log = open(out + '.log', 'w')
        best_error = np.inf
        for epoch in tqdm.tqdm(range(epochs)):
            for i, batch in enumerate(loader_train):
                optimizer.zero_grad()
                loss_train = self.likelihood(batch.to(utils.dev))
                loss_train = -loss_train.mean()
                loss_train.backward()
                optimizer.step()
                if i % 10 == 0 or loader_valid is not None:
                    seqs = self.sample(len(batch * 2))
                    ix = utils.unique(seqs)
                    seqs = seqs[ix]
                    smiles, valids = self.voc.check_smiles(seqs)
                    error = 1 - sum(valids) / len(seqs)
                    info = "Epoch: %d step: %d error_rate: %.3f loss_train: %.3f" % (epoch, i, error, loss_train.item())
                    if loader_valid is not None:
                        loss_valid, size = 0, 0
                        for j, batch in enumerate(loader_valid):
                            size += batch.size(0)
                            loss_valid += -self.likelihood(batch.to(utils.dev)).sum().item()
                        loss_valid = loss_valid / size / self.voc.max_len
                        if loss_valid < best_error:
                            torch.save(self.state_dict(), out + '.pkg')
                            best_error = loss_valid
                        info += ' loss_valid: %.3f' % loss_valid
                    elif error < best_error:
                        torch.save(self.state_dict(), out + '.pkg')
                        best_error = error
                    print(info, file=log)
                    for i, smile in enumerate(smiles):
                        print('%d\t%s' % (valids[i], smile), file=log)
        log.close()

class Generator(nn.Module):
    """
    堆叠循环神经网络模型，用于生成SMILES
    """
    def __init__(self, voc, embed_size=128, hidden_size=512, num_layers=3, is_lstm=True, lr=1e-3):
        super(Generator, self).__init__()
        self.voc = voc
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = voc.size
        #RNN堆叠层数
        self.num_layers = num_layers
        #嵌入层
        self.embed = nn.Embedding(voc.size, embed_size)
        self.is_lstm = is_lstm
        rnn_layer = nn.LSTM if is_lstm else nn.GRU
        self.rnn = nn.ModuleList()
        self.rnn.append(rnn_layer(embed_size, hidden_size, num_layers=3, batch_first=True))
        for i in range(num_layers-1):
            self.rnn.append(rnn_layer(embed_size, hidden_size, num_layers=3, batch_first=True))
        #self.attention = nn.MultiheadAttention(hidden_size, embed_size)
        self.linear = nn.Linear(hidden_size, voc.size)
        self.fixlinear = nn.Linear(512, 128)
        self.activation = nn.ReLU()
        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.to(utils.dev)

    def forward(self, input, h):
        output = self.embed(input.unsqueeze(-1))
        output, h_out = self.rnn[0](output, h)
        input_t = self.fixlinear(output)
        for i in range(1, self.num_layers):
            output, h_out = self.rnn[i](input_t, h_out)
            #格式对齐
            input_t = self.fixlinear(output)
            # 添加激活函数
            input_t = self.activation(input_t)
        output = self.linear(output).squeeze(1)
        return output, h_out

    def init_h(self, batch_size, labels=None):
        h = torch.rand(3, batch_size, 512).to(utils.dev)
        if labels is not None:
            h[:, :, 0] = labels
        if self.is_lstm:
            c = torch.rand(3, batch_size, self.hidden_size).to(utils.dev)
            return (h, c)
        else:
            return h

    def likelihood(self, target):
        batch_size, seq_len = target.size()
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(utils.dev)
        h = self.init_h(batch_size)  # 初始化隐藏状态列表
        scores = torch.zeros(batch_size, seq_len).to(utils.dev)
        for step in range(seq_len):
            logit, h = self(x, h)
            logit = logit.log_softmax(dim=-1)
            score = logit.gather(1, target[:, step:step+1]).squeeze()
            scores[:, step] = score
            x = target[:, step]
        return scores

    def PGLoss(self, loader):
        for seq, reward in loader:
            self.zero_grad()
            score = self.likelihood(seq)
            loss = score * reward
            loss = -loss.mean()
            loss.backward()
            self.optim.step()

    def sample(self, batch_size):
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(utils.dev)
        h = self.init_h(batch_size)  # 初始化所有层的 hidden state
        sequences = torch.zeros(batch_size, self.voc.max_len).long().to(utils.dev)
        isEnd = torch.zeros(batch_size).bool().to(utils.dev)

        for step in range(self.voc.max_len):
            logits, hs = self(x, h)  # 计算每一层的输出和新的 hidden state
            h = hs
            proba = logits.softmax(dim=-1)
            x = torch.multinomial(proba, 1).view(-1)
            x[isEnd] = self.voc.tk2ix['EOS']
            sequences[:, step] = x

            end_token = (x == self.voc.tk2ix['EOS'])
            isEnd = torch.ge(isEnd + end_token, 1)
            if (isEnd == 1).all(): break
        return sequences

    def evolve(self, batch_size, epsilon=0.01, crover=None, mutate=None):
        # Start tokens
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(utils.dev)
        # Hidden states initialization for exploitation network
        h = self.init_h(batch_size)
        # Hidden states initialization for exploration network
        h1 = self.init_h(batch_size)
        h2 = self.init_h(batch_size)
        # Initialization of output matrix
        sequences = torch.zeros(batch_size, self.voc.max_len).long().to(utils.dev)
        # labels to judge and record which sample is ended
        is_end = torch.zeros(batch_size).bool().to(utils.dev)

        for step in range(self.voc.max_len):
            logit, h = self(x, h)
            proba = logit.softmax(dim=-1)
            if crover is not None:
                ratio = torch.rand(batch_size, 1).to(utils.dev)
                logit1, h1 = crover(x, h1)
                proba = proba * ratio + logit1.softmax(dim=-1) * (1 - ratio)
            if mutate is not None:
                logit2, h2 = mutate(x, h2)
                is_mutate = (torch.rand(batch_size) < epsilon).to(utils.dev)
                proba[is_mutate, :] = logit2.softmax(dim=-1)[is_mutate, :]
            # sampling based on output probability distribution
            x = torch.multinomial(proba, 1).view(-1)

            is_end |= x == self.voc.tk2ix['EOS']
            x[is_end] = self.voc.tk2ix['EOS']
            sequences[:, step] = x
            if is_end.all(): break
        return sequences

    def evolve1(self, batch_size, epsilon=0.01, crover=None, mutate=None):
        # Start tokens
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(utils.dev)
        # Hidden states initialization for exploitation network
        h = self.init_h(batch_size)
        # Hidden states initialization for exploration network
        h2 = self.init_h(batch_size)
        # Initialization of output matrix
        sequences = torch.zeros(batch_size, self.voc.max_len).long().to(utils.dev)
        # labels to judge and record which sample is ended
        is_end = torch.zeros(batch_size).bool().to(utils.dev)

        for step in range(self.voc.max_len):
            is_change = torch.rand(1) < 0.5
            if crover is not None and is_change:
                logit, h = crover(x, h)
            else:
                logit, h = self(x, h)
            proba = logit.softmax(dim=-1)
            if mutate is not None:
                logit2, h2 = mutate(x, h2)
                ratio = torch.rand(batch_size, 1).to(utils.dev) * epsilon
                proba = logit.softmax(dim=-1) * (1 - ratio) + logit2.softmax(dim=-1) * ratio
            # sampling based on output probability distribution
            x = torch.multinomial(proba, 1).view(-1)

            x[is_end] = self.voc.tk2ix['EOS']
            sequences[:, step] = x

            # Judging whether samples are end or not.
            end_token = (x == self.voc.tk2ix['EOS'])
            is_end = torch.ge(is_end + end_token, 1)
            #  If all of the samples generation being end, stop the sampling process
            if (is_end == 1).all(): break
        return sequences

    def fit(self, loader_train, out, loader_valid=None, epochs=100, lr=1e-3):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        log = open(out + '.log', 'w')
        best_error = np.inf
        for epoch in tqdm.tqdm(range(epochs)):
            for i, batch in enumerate(loader_train):
                optimizer.zero_grad()
                loss_train = self.likelihood(batch.to(utils.dev))
                loss_train = -loss_train.mean()
                loss_train.backward()
                optimizer.step()
                if i % 10 == 0 or loader_valid is not None:
                    seqs = self.sample(len(batch * 2))
                    ix = utils.unique(seqs)
                    seqs = seqs[ix]
                    smiles, valids = self.voc.check_smiles(seqs)
                    error = 1 - sum(valids) / len(seqs)
                    info = "Epoch: %d step: %d error_rate: %.3f loss_train: %.3f" % (epoch, i, error, loss_train.item())
                    if loader_valid is not None:
                        loss_valid, size = 0, 0
                        for j, batch in enumerate(loader_valid):
                            size += batch.size(0)
                            loss_valid += -self.likelihood(batch.to(utils.dev)).sum().item()
                        loss_valid = loss_valid / size / self.voc.max_len
                        if loss_valid < best_error:
                            torch.save(self.state_dict(), out + '.pkg')
                            best_error = loss_valid
                        info += ' loss_valid: %.3f' % loss_valid
                    elif error < best_error:
                        torch.save(self.state_dict(), out + '.pkg')
                        best_error = error
                    print(info, file=log)
                    for i, smile in enumerate(smiles):
                        print('%d\t%s' % (valids[i], smile), file=log)
        log.close()

class Atten(nn.Module):
    """
    添加注意力机制
    """
    def __init__(self, voc, embed_size=128, hidden_size=512, num_layers=3, is_lstm=True, lr=1e-3):
        super(Atten, self).__init__()
        self.voc = voc
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = voc.size
        #RNN堆叠层数
        self.num_layers = num_layers
        #嵌入层
        self.embed = nn.Embedding(voc.size, embed_size)
        self.is_lstm = is_lstm
        rnn_layer = nn.LSTM if is_lstm else nn.GRU
        self.rnn = nn.ModuleList()
        self.rnn.append(rnn_layer(embed_size, hidden_size, num_layers=3, batch_first=True))
        for i in range(num_layers-1):
            self.rnn.append(rnn_layer(embed_size, hidden_size, num_layers=3, batch_first=True))
        self.attention = nn.MultiheadAttention(hidden_size, embed_size)
        self.linear = nn.Linear(hidden_size, voc.size)
        self.fixlinear = nn.Linear(512, 128)
        self.activation = nn.ReLU()
        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.to(utils.dev)

    def forward(self, input, h):
        output = self.embed(input.unsqueeze(-1))
        output, h_out = self.rnn[0](output, h)
        input_t = self.fixlinear(output)
        for i in range(1, self.num_layers):
            output, h_out = self.rnn[i](input_t, h_out)
            input_t = self.activation(output)
            input_t = self.fixlinear(input_t)
        output = self.linear(output).squeeze(1)
        return output, h_out

    def init_h(self, batch_size, labels=None):
        h = torch.rand(3, batch_size, 512).to(utils.dev)
        if labels is not None:
            h[:, :, 0] = labels
        if self.is_lstm:
            c = torch.rand(3, batch_size, self.hidden_size).to(utils.dev)
            return (h, c)
        else:
            return h

    def likelihood(self, target):
        batch_size, seq_len = target.size()
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(utils.dev)
        h = self.init_h(batch_size)  # 初始化隐藏状态列表
        scores = torch.zeros(batch_size, seq_len).to(utils.dev)
        for step in range(seq_len):
            logit, h = self(x, h)
            logit = logit.log_softmax(dim=-1)
            score = logit.gather(1, target[:, step:step+1]).squeeze()
            scores[:, step] = score
            x = target[:, step]
        return scores

    def PGLoss(self, loader):
        for seq, reward in loader:
            self.zero_grad()
            score = self.likelihood(seq)
            loss = score * reward
            loss = -loss.mean()
            loss.backward()
            self.optim.step()

    def sample(self, batch_size):
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(utils.dev)
        h = self.init_h(batch_size)  # 初始化所有层的 hidden state
        sequences = torch.zeros(batch_size, self.voc.max_len).long().to(utils.dev)
        isEnd = torch.zeros(batch_size).bool().to(utils.dev)

        for step in range(self.voc.max_len):
            logits, hs = self(x, h)  # 计算每一层的输出和新的 hidden state
            h = hs
            proba = logits.softmax(dim=-1)
            x = torch.multinomial(proba, 1).view(-1)
            x[isEnd] = self.voc.tk2ix['EOS']
            sequences[:, step] = x

            end_token = (x == self.voc.tk2ix['EOS'])
            isEnd = torch.ge(isEnd + end_token, 1)
            if (isEnd == 1).all(): break
        return sequences

    def evolve(self, batch_size, epsilon=0.01, crover=None, mutate=None):
        # Start tokens
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(utils.dev)
        # Hidden states initialization for exploitation network
        h = self.init_h(batch_size)
        # Hidden states initialization for exploration network
        h1 = self.init_h(batch_size)
        h2 = self.init_h(batch_size)
        # Initialization of output matrix
        sequences = torch.zeros(batch_size, self.voc.max_len).long().to(utils.dev)
        # labels to judge and record which sample is ended
        is_end = torch.zeros(batch_size).bool().to(utils.dev)

        for step in range(self.voc.max_len):
            logit, h = self(x, h)
            proba = logit.softmax(dim=-1)
            if crover is not None:
                ratio = torch.rand(batch_size, 1).to(utils.dev)
                logit1, h1 = crover(x, h1)
                proba = proba * ratio + logit1.softmax(dim=-1) * (1 - ratio)
            if mutate is not None:
                logit2, h2 = mutate(x, h2)
                is_mutate = (torch.rand(batch_size) < epsilon).to(utils.dev)
                proba[is_mutate, :] = logit2.softmax(dim=-1)[is_mutate, :]
            # sampling based on output probability distribution
            x = torch.multinomial(proba, 1).view(-1)

            is_end |= x == self.voc.tk2ix['EOS']
            x[is_end] = self.voc.tk2ix['EOS']
            sequences[:, step] = x
            if is_end.all(): break
        return sequences

    def evolve1(self, batch_size, epsilon=0.01, crover=None, mutate=None):
        # Start tokens
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(utils.dev)
        # Hidden states initialization for exploitation network
        h = self.init_h(batch_size)
        # Hidden states initialization for exploration network
        h2 = self.init_h(batch_size)
        # Initialization of output matrix
        sequences = torch.zeros(batch_size, self.voc.max_len).long().to(utils.dev)
        # labels to judge and record which sample is ended
        is_end = torch.zeros(batch_size).bool().to(utils.dev)

        for step in range(self.voc.max_len):
            is_change = torch.rand(1) < 0.5
            if crover is not None and is_change:
                logit, h = crover(x, h)
            else:
                logit, h = self(x, h)
            proba = logit.softmax(dim=-1)
            if mutate is not None:
                logit2, h2 = mutate(x, h2)
                ratio = torch.rand(batch_size, 1).to(utils.dev) * epsilon
                proba = logit.softmax(dim=-1) * (1 - ratio) + logit2.softmax(dim=-1) * ratio
            # sampling based on output probability distribution
            x = torch.multinomial(proba, 1).view(-1)

            x[is_end] = self.voc.tk2ix['EOS']
            sequences[:, step] = x

            # Judging whether samples are end or not.
            end_token = (x == self.voc.tk2ix['EOS'])
            is_end = torch.ge(is_end + end_token, 1)
            #  If all of the samples generation being end, stop the sampling process
            if (is_end == 1).all(): break
        return sequences

    def fit(self, loader_train, out, loader_valid=None, epochs=100, lr=1e-3):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        log = open(out + '.log', 'w')
        best_error = np.inf
        for epoch in tqdm.tqdm(range(epochs)):
            for i, batch in enumerate(loader_train):
                optimizer.zero_grad()
                loss_train = self.likelihood(batch.to(utils.dev))
                loss_train = -loss_train.mean()
                loss_train.backward()
                optimizer.step()
                if i % 10 == 0 or loader_valid is not None:
                    seqs = self.sample(len(batch * 2))
                    ix = utils.unique(seqs)
                    seqs = seqs[ix]
                    smiles, valids = self.voc.check_smiles(seqs)
                    error = 1 - sum(valids) / len(seqs)
                    info = "Epoch: %d step: %d error_rate: %.3f loss_train: %.3f" % (epoch, i, error, loss_train.item())
                    if loader_valid is not None:
                        loss_valid, size = 0, 0
                        for j, batch in enumerate(loader_valid):
                            size += batch.size(0)
                            loss_valid += -self.likelihood(batch.to(utils.dev)).sum().item()
                        loss_valid = loss_valid / size / self.voc.max_len
                        if loss_valid < best_error:
                            torch.save(self.state_dict(), out + '.pkg')
                            best_error = loss_valid
                        info += ' loss_valid: %.3f' % loss_valid
                    elif error < best_error:
                        torch.save(self.state_dict(), out + '.pkg')
                        best_error = error
                    print(info, file=log)
                    for i, smile in enumerate(smiles):
                        print('%d\t%s' % (valids[i], smile), file=log)
        log.close()



