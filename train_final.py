
import conllu
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# from 'dataset.py' import POSTaggingDataset

device = "cuda" #if torch.cuda.is_available() else "cpu"

with open('UD_English-Atis/en_atis-ud-train.conllu', 'r', encoding="utf-8") as f:
    sentences = conllu.parse(f.read())

    train_data = []

    for sentence in sentences:
        indexes = []
        words = []
        pos = []
        # print(sentence)
        for i in range(len(sentence)):
            indexes.append(sentence[i]['id'])
            words.append(sentence[i]['form'])
            pos.append(sentence[i]['upos'])
        tagged_sentence = (words, pos)
        train_data.append(tagged_sentence)
# print(train_data)

tag_to_idx = {
    "ADJ": 0,
    "ADP": 1,
    "ADV": 2,
    "AUX": 3,
    "CCONJ": 4,
    "DET": 5,
    "INTJ": 6,
    "NOUN": 7,
    "NUM": 8,
    "PART": 9,
    "PRON": 10,
    "PROPN": 11,
    "PUNCT": 12,
    "SCONJ": 13,
    "SYM": 14,
    "VERB": 15,
    "X": 16
}

tagset_size = len(tag_to_idx)

# To create word embeddings, we need to create a vocabulary of words.
# We will use the vocabulary to create a mapping from words to indices.
word_to_idx = {}
for sentence, tags in train_data:
    for word in sentence:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
vocab_size = len(word_to_idx)

"""# MODEL

"""

embedding_dim = 128
hidden_dim = 128

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size  ########
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 
            num_layers=1)
        self.tagset_size = tagset_size  ########
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

"""# TRAIN THE MODEL"""

model = LSTMTagger(embedding_dim, hidden_dim, vocab_size, tagset_size).to(device)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.18)   

def input_seq(sentence, word_to_idx):
    idxs = []
    for w in sentence:
        try:
            if w in word_to_idx:
                idxs.append(word_to_idx[w])
        except:
            continue
        return torch.tensor(idxs, dtype=torch.long).to(device)

#  Doubt, if a sequence wiht entirely new words is given, i dont think it will be able to predict the tags

for epoch in range(15): 
    print("Epoch: ", epoch)
    print("Device type = ", device)
    for sentence, tags in train_data:
        model.zero_grad()
        input_sentence = input_seq(sentence, word_to_idx)
        # print(len(input_sentence))
        targets = input_seq(tags, tag_to_idx)
        # print(len(targets)
        tag_scores = model(input_sentence).to(device)
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
    print("Loss: ", loss.item())
    #calculate accuracy
    correct_counts = 0
    total_counts = 0
    with torch.no_grad():
        for sentence, tags in train_data:
            input_sentence = input_seq(sentence, word_to_idx)
            # print(len(input_sentence))
            targets = input_seq(tags, tag_to_idx)
            # print(len(targets))
            tag_scores = model(input_sentence).to(device)
            #print("Tag scores = ", len(tag_scores))
            for i in range(len(tag_scores)):
                predicted_index = torch.argmax(tag_scores[i])
                if predicted_index == targets[i]:
                    correct_counts += 1
                total_counts += 1
        print("Training Accuracy = ", correct_counts/total_counts)
        

# torch.save(model.state_dict(), "model.pt")

"""# TEST THE MODEL"""


# torch.load("model.pt")
# model.eval()

with open('UD_English-Atis/en_atis-ud-dev.conllu', 'r', encoding="utf-8") as f:
    sentences = conllu.parse(f.read())

test_data = []

for sentence in sentences:
    indexes = []
    words = []
    pos = []
    for i in range(len(sentence)):
        indexes.append(sentence[i]['id'])
        words.append(sentence[i]['form'])
        pos.append(sentence[i]['upos'])
    tagged_sentence = (words, pos)
    test_data.append(tagged_sentence)

correct_counts = 0
total_counts = 0

with torch.no_grad():
        for sentence, tags in test_data:
            input_sentence = input_seq(sentence, word_to_idx)
            targets = input_seq(tags, tag_to_idx)
            tag_scores = model(input_sentence).to(device)
            for i in range(len(tag_scores)):
                predicted_index = torch.argmax(tag_scores[i])
                if predicted_index == targets[i]:
                    correct_counts += 1
                total_counts += 1

print("Validation Accuracy = ", correct_counts/total_counts)















# while(1):
#     new_sentence = input("Enter a sentence: ")
#     new_sentence = new_sentence.lower()
#     new_sentence = new_sentence.strip()
#     with torch.no_grad():
#         new_input = input_seq(new_sentence.split(), word_to_idx)
#         tag_scores = model(new_input)
#         for i in range(len(tag_scores)):
#                 # print(tag_scores[i].argmax().item())
#             for key, value in tag_to_idx.items():
#                     if value == tag_scores[i].argmax().item():
#                         print(key)
#                         break
        