from platform import release
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModel(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.LSTM_hidden = args.LSTM_hidden
        self.MLP_hidden = args.MLP_hidden
        self.attention_hidden = args.attention_hidden
        self.aspects = args.aspects
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size
        
        
        self.LSTM_part = nn.LSTM(100, self.LSTM_hidden, 1, batch_first=True, bidirectional=True)

        self.attention = nn.Sequential(
            nn.Linear(self.LSTM_hidden * 2, self.attention_hidden),
            nn.Tanh(),
            nn.Linear(self.attention_hidden, self.aspects)
        )

        self.cls = nn.Sequential(
            nn.Linear(self.aspects * self.LSTM_hidden * 2, self.MLP_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.MLP_hidden, self.num_classes),
        )
        
    
    def forward(self, x, retain_A=False):
        # x: [sentence_len, word_embedding_dimension]
        # H: [sentence_len, LSTM_hidden * 2]
        H, _ = self.LSTM_part(x)

        A = self.attention(H)
        A = F.softmax(A, dim=1) # 1 or 2
        
        M = torch.bmm(torch.permute(A, (0, 2, 1)), H)   # A.T or torch.permute(A, (0, 2, 1))
        M = torch.flatten(M, 1)

        out = self.cls(M)
        if retain_A:
            return out, A
        else:
            return out



class BiLSTM(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.LSTM_hidden = args.LSTM_hidden
        self.MLP_hidden = args.MLP_hidden
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size
        
        
        self.LSTM_part = nn.LSTM(100, self.LSTM_hidden, 1, batch_first=True, bidirectional=True)

        self.cls = nn.Sequential(
            nn.Linear(self.LSTM_hidden * 2, self.MLP_hidden),
            nn.ReLU(),
            nn.Linear(self.MLP_hidden, self.num_classes),
        )
        
    
    def forward(self, x):
        # x: [sentence_len, word_embedding_dimension]
        # H: [sentence_len, LSTM_hidden * 2]
        H, _ = self.LSTM_part(x)
        H = H.max(1).values
        out = self.cls(H)
        return out


class CNNModel(nn.Module):
    
    def __init__(self, args):
        super().__init__()

        self.word_embedding_dimension = args.word_embedding_dimension
        self.out_channels = args.out_channels
        self.kernel_size = args.kernel_size
        self.MLP_hidden = args.MLP_hidden
        self.num_classes = args.num_classes

        self.CNN = nn.Conv1d(self.word_embedding_dimension, self.out_channels, self.kernel_size)
        self.cls = nn.Sequential(
            nn.Linear(self.out_channels, self.MLP_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.MLP_hidden, self.num_classes)
        )

    def forward(self, x):
        x = self.CNN(x.permute((0, 2, 1)))
        x = x.max(dim=2).values
        x = self.cls(x)
        return x
