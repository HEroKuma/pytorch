import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """There's a lady who's sure
All that glitters is gold
And she's buying a stairway to heaven
When she gets there she knows
If the stores are all closed
With a word she can get what she came for
Oh oh oh oh and she's buying a stairway to heaven
There's a sign on the wall
But she wants to be sure
'Cause you know sometimes words have two meanings
In a tree by the brook
There's a songbird who sings
Sometimes all of our thoughts are misgiving
Ooh, it makes me wonder
Ooh, it makes me wonder
There's a feeling I get
When I look to the west
And my spirit is crying for leaving
In my thoughts I have seen
Rings of smoke through the trees
And the voices of those who standing looking
Ooh, it makes me wonder
Ooh, it really makes me wonder
And it's whispered that soon, If we all call the tune
Then the piper will lead us to reason
And a new day will dawn
For those who stand long
And the forests will echo with laughter
If there's a bustle in your hedgerow
Don't be alarmed now
It's just a spring clean for the May queen
Yes, there are two paths you can go by
But in the long run
There's still time to change the road you're on
And it makes me wonder
Your head is humming and it won't go
In case you don't know
The piper's calling you to join him
Dear lady, can you hear the wind blow
And did you know
Your stairway lies on the whispering wind
And as we wind on down the road
Our shadows taller than our soul
There walks a lady we all know
Who shines white light and wants to show
How everything still turns to gold
And if you listen very hard
The tune will come to you at last
When all are one and one is all
To be a rock and not to roll
And she's buying the stairway to heaven""".split()

trigram = [((test_sentence[i], test_sentence[i + 1]), test_sentence[i + 2])
           for i in range(len(test_sentence) - 2)]

vocb = set(test_sentence)
word_to_idx = {word: i for i, word in enumerate(vocb)}
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}


class NgramModel(nn.Module):
    def __init__(self, vocb_size, context_size, n_dim):
        super(NgramModel, self).__init__()
        self.n_word = vocb_size
        self.embedding = nn.Embedding(self.n_word, n_dim)
        self.linear1 = nn.Linear(context_size * n_dim, 128)
        self.linear2 = nn.Linear(128, self.n_word)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(1, -1)
        out = self.linear1(emb)
        out = F.relu(out)
        out = self.linear2(out)
        log_prob = F.log_softmax(out)
        return log_prob


ngrammodel = NgramModel(len(word_to_idx), CONTEXT_SIZE, 100)
criterion = nn.NLLLoss()
optimizer = optim.SGD(ngrammodel.parameters(), lr=1e-3)

for epoch in range(100):
    print('epoch: {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0
    for data in trigram:
        word, label = data
        word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
        label = Variable(torch.LongTensor([word_to_idx[label]]))
        # forward
        out = ngrammodel(word)
        loss = criterion(out, label)
        running_loss += loss.data[0]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Loss: {:.6f}'.format(running_loss / len(word_to_idx)))

word, label = trigram[3]
word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
out = ngrammodel(word)
_, predict_label = torch.max(out, 1)
print(predict_label.data)
predict_word = idx_to_word[predict_label.data[0]]
print('real word is {}, predict word is {}'.format(label, predict_word))