# Team ID - Hw2Grp
# Team Members -
# Kovid Sharma (z5240067)
# Vidyadhari Prerepa (z5284443)

#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""


#Our approach (Answer to the Question):
'''
The program begins with tokenizing where each review is tokenized into separate words. This 
is followed by pre-processing of the reviews. In this step, we remove illegal characters such as "\/*?:"<>|"
from the reviews as these have no effect on the rating or business category. This is followed by the removal 
of numbers and punctuations such as 'the', 'and', 'an' etc using standard python libraries. We have changed
the word vector dimension from default 50 to 300 which was found to be much more efficient for the network.

We've made a new function 'cleanse' which only keeps ascii.string characters, everything else is discarded.
We also tried limiting the number of letters in a word, eg only keep words more than a length of two which
would mean discarding words like 'i', 'me', 'it', 'we' etc. Also, all text is converted to lowercase.

The convertNetOutput method is used to convert both the rating and business category to long in cases when the
network outputs a result of different representation. For the network model, we initially decided to do 
LSTM. We then tried bi-LSTM, bi-GRU, bi-GRU with glove vectors and finally decided to with bidirectional LSTM.
The network has two hidden layers which converge to give two different types of outputs, one with number of 
outputs = 2 (rating) and another with number of outputs = 5 (business category).
The forward() function initially concatenates the output of original order and its reversed order over dimension 1,
followed by Relu activation function and finally squeezing of the output to remove single-dimensional entries 
from the network output tensor. We also tried various combinations switching F.log_softmax on and off based on the
loss function that we we're using. It turned out best to used it for MultiMarginLoss. It wasn't required for 
CrossEntropyLoss since it automatically applies log_softmax with NLLLoss.
We're setting dropout to 0.5. dropout - if non-zero, introduces a Dropout layer on the outputs of each LSTM 
layer except the last layer, with dropout probability equal to dropout. By Default it is set to 0. 
We're changing it to 0.5 which means creating a dropout layer with a 50% chance of setting inputs to zero.

For the loss function, we have used CrossEntropyLoss which was found to be much better for predicting 
rating. Our initial approach was to use BCELoss or BCEWithLogitsLoss since rating is a binary prediction. 
Eventually, CrossEntropyLoss resulted in much accurate outputs for rating as well. We've used 
MultiMarginLoss for category with log_softmax because it creates a criterion that optimizes 
multi-class classification hinge loss (margin-based loss) between input and output.

We have set the trainValSplit to the default value that is 0.8. However we made changes to the batch size and 
learning rate. Initially we had increased the epochs to 20, learning rate to 0.005 and batch size to 64 which
resulted in an average weighted score of 82.64. This was improved to 86.26 when epochs were set to 20 
and learning rate was 0.003 with a batch size of 128. We have tried a variety of optimizers including
Adam, SGD, AdamW and Adadelta. The network converged faster and gave much lesser loss and higher weighted 
average score with Adam. Adadelta performed the worst of all.

AdamW
In the common weight decay implementation in the Adam optimizer the weight decay is implicitly bound to the 
learning rate. This means that when optimizing the learning rate you will also need to find a new optimal 
weight decay for each learning rate you try. The AdamW optimizer decouples the weight decay from the 
optimization step. This means that the weight decay and learning rate can be optimized separately, i.e. 
changing the learning rate does not change the optimal weight decay. The result of this fix is a substantially 
improved generalization performance

For the glove vectors we experimented with dimension 50,100,200 and 300. The difference between 100, 200 and 300
was marginal but with 50 dimensions, the result was quite poor. In fact, after rigours testing it was clear that the
dimensions should be 300. To generalize the model, we converted everything to lowercase, removed the junk chars and 
numbers, got rid of any hyperlinks, ran the model for 20 epochs, trained with a batchsize of 128 so that a lot of
examples of all categories enter the network in one try and not bias towards any category.

We also tried nltk porter stemmer, snowball stemmer, lemmatization but they all decreased the performance considerably.
We also made an attempt at batch normalization but even that didn't improve the performance above 86.26%.
'''


import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as toptim
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn
import string
import re

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################


def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split()

    return processed


# a function to keep only ascii letters [a-zA-Z]
def cleanse(dirty_text):
    dirty_text = ' '.join(dirty_text)
    dirty_text = dirty_text.lower()
    alphanumeric = ''
    # remove urls
    url_temp = re.findall(r'(https?://\S+)', dirty_text)
    if url_temp != []:
        for e in url_temp:
            dirty_text = dirty_text.replace(e, ' ')

    for character in dirty_text:
        if character in string.ascii_letters:
            alphanumeric += character
        else:
            alphanumeric += ' '

    # temp = alphanumeric.split()
    # sample_text = ''
    # for e in temp:
    #     if len(e) >= 2:  # keep words more than 2 letters
    #         sample_text += e + ' '
    # sample = sample_text.strip()
    # sample = sample.split()
    sample = alphanumeric.split()
    return sample


def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    sample = cleanse(sample)
    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """
    return batch

stopWords = {}

wordVectorDimension = 300
wordVectors = GloVe(name='6B', dim=wordVectorDimension)


################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
    ratingOutput = (torch.argmax(ratingOutput, 1)).long()
    categoryOutput = (torch.argmax(categoryOutput, 1)).long()
    return ratingOutput, categoryOutput


################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()
        self.lstm = tnn.LSTM(wordVectorDimension, hidden_dim, num_layers=2,
                             batch_first=True, bidirectional=True, dropout=0.5)
        # self.gru = tnn.GRU(wordVectorDimension, hidden_dim, num_layers=2,
        #                    batch_first=True, bidirectional=True, dropout=0.5)
        self.fc1 = tnn.Linear(hidden_dim*4, fc_dim)
        self.fc2 = tnn.Linear(fc_dim, 2)
        self.fc3 = tnn.Linear(fc_dim, 5)

    def forward(self, input, length):
        out, (h_n, c_n) = self.lstm(input)
        # out, hidden = self.gru(input)
        # concatenate the output of normal order and reversed order
        x = torch.cat((out[:, -1, :], out[:, 0, :]), dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x1 = self.fc2(x)
        # x1 = F.log_softmax(x1, dim=1)
        rating = x1.squeeze()

        y1 = self.fc3(x)
        y1 = F.log_softmax(y1, dim=1)
        category = y1.squeeze()
        return rating, category


class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """
    def __init__(self):
        super(loss, self).__init__()
        self.lossrating = tnn.CrossEntropyLoss()
        self.losscategory = tnn.MultiMarginLoss()
        # self.losscategory = tnn.CrossEntropyLoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        ratingTarget = ratingTarget.long()
        ratingresult = self.lossrating(ratingOutput, ratingTarget)

        categoryTarget = categoryTarget.long()
        categoryresult = self.losscategory(categoryOutput, categoryTarget)
        return ratingresult+categoryresult


hidden_dim = 100
fc_dim = 200

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 128
epochs = 20
lrate = 0.003

# faster converge using Adam than SGD
# optimiser = toptim.SGD(net.parameters(), lr=0.7)
optimiser = toptim.Adam(net.parameters(), lr=lrate)
# optimiser = toptim.AdamW(net.parameters(), lr=lrate)
# optimiser = toptim.Adadelta(net.parameters(), lr=lrate)

