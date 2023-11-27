import math
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


# class Classifier(nn.Module):
#     def __init__(self, hidden_size):
#         super(Classifier, self).__init__()
#         self.linear1 = nn.Linear(hidden_size, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, mask_cls):
#         h = self.linear1(x).squeeze(-1)
#         sent_scores = self.sigmoid(h) * mask_cls.float()
#         return sent_scores

# class Classifier(nn.Module):
#     def __init__(self, hidden_size, dropout_rate):
#         super(Classifier, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.BatchNorm1d(hidden_size),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.BatchNorm1d(hidden_size), 
#             nn.Linear(hidden_size, 1)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, mask_cls):
#         x = self.layers(x).squeeze(-1)
#         sent_scores = self.sigmoid(x) * mask_cls.float()
#         return sent_scores

