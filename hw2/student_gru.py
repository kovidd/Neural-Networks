import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as toptim
from torchtext.vocab import GloVe
import string as s
import re as r


def tokenise(sample):
    processed = sample.split()
    return processed


def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    # remove illegal characters
    sample = [r.sub(r'[^\x00-\x7f]', r'', w) for w in sample]
    # remove punctuations
    sample = [w.strip(s.punctuation) for w in sample]
    # remove numbers
    sample = [r.sub(r'[0-9]', r'', w) for w in sample]
    return sample


def postprocessing(batch, vocab):
    # vocab = 1
    return batch


stopWords = {}

wordVectorDimension = 300
wordVectors = GloVe(name='6B', dim=wordVectorDimension)
hidden_dim = 100
fc_dim = 72


def convertNetOutput(ratingOutput, categoryOutput):
    ratingOutput = (torch.argmax(ratingOutput, 1)).long()
    categoryOutput = (torch.argmax(categoryOutput, 1)).long()
    return ratingOutput, categoryOutput



class network(tnn.Module):
    def __init__(self, hidden_size, vocab_size, n_extra_feat, weights_matrix, output_size, num_layers=2, dropout=0.2,
                 spatial_dropout=True, bidirectional=True):
        super(network, self).__init__()
        # Initialize attributes
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_extra_feat = n_extra_feat
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.spatial_dropout = spatial_dropout
        self.bidirectional = bidirectional
        self.n_directions = 2 if self.bidirectional else 1
        self.batch_size = 128

        self.gru = tnn.GRU(wordVectorDimension, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
        self.fc1 = tnn.Linear(hidden_dim*4, fc_dim)
        self.fc2 = tnn.Linear(fc_dim, 2)
        self.fc3 = tnn.Linear(fc_dim, 5)

    def forward(self, input, length):
        gru_out, hidden = self.gru(input)
        # hidden = hidden.view(4, self.n_directions, self.batch_size, self.hidden_size)
        last_hidden = hidden[-1]
        last_hidden = torch.sum(last_hidden, dim=0)
        gru_out, lengths = tnn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        if self.bidirectional:
            gru_out = gru_out[:, :, :self.hidden_size] + gru_out[:, :, self.hidden_size:]
        max_pool = F.adaptive_max_pool1d(gru_out.permute(0, 2, 1), (1,)).view(self.batch_size, -1)
        avg_pool = torch.sum(gru_out, dim=1) / lengths.view(-1, 1).type(torch.FloatTensor)

        concat_out = torch.cat([last_hidden, max_pool, avg_pool, input], dim=1)
        out = self.linear(concat_out)


        x = torch.cat((out[:, -1, :], out[:, 0, :]), dim=1)
        x = self.fc1(x)
        x = tnn.functional.relu(x)
        x1 = self.fc2(x)
        x1 = F.log_softmax(x1, dim=1)
        rating = x1.squeeze()

        y1 = self.fc3(x)
        y1 = F.log_softmax(y1, dim=1)
        category = y1.squeeze()
        return rating, category


class loss(tnn.Module):
    def __init__(self):
        super(loss, self).__init__()
        # self.lossrating = tnn.CrossEntropyLoss()
        # self.losscategory = tnn.CrossEntropyLoss()
        self.lossrating = tnn.MultiMarginLoss()
        self.losscategory = tnn.MultiMarginLoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        ratingTarget = ratingTarget.long()
        ratingresult = self.lossrating(ratingOutput, ratingTarget)

        categoryTarget = categoryTarget.long()
        categoryresult = self.losscategory(categoryOutput, categoryTarget)
        return ratingresult+categoryresult

# Initialize parameters
hidden_size = 8
vocab_size = 300
n_extra_feat = 10
output_size = 2
num_layers = 2
dropout = 0.5
learning_rate = 0.001
epochs = 40
spatial_dropout = True

net = network(hidden_size, vocab_size, n_extra_feat, output_size, num_layers, dropout,
              spatial_dropout, bidirectional=True)
lossFunc = loss()


# more training data than default ratio
trainValSplit = 0.8
batchSize = 128
epochs = 1
lrate = 0.009

# faster converge using Adam than SGD
# optimiser = toptim.SGD(net.parameters(), lr=0.7)
optimiser = toptim.Adam(net.parameters(), lr=lrate)
# optimiser = toptim.AdamW(net.parameters(), lr=lrate)
# optimiser = toptim.Adadelta(net.parameters(), lr=lrate)