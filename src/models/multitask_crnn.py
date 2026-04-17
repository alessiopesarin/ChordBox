import torch
import torch.nn as nn

class MultiTaskChordCRNN(nn.Module):
    """
    Multi-Task Architecture: Predicts Root and Quality separately.
    """
    def __init__(self, num_roots=13, num_quals=6):
        super(MultiTaskChordCRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((1, 2)), 
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((1, 2))
        )
        # 128 channels * 3 frequency bins = 384
        self.rnn = nn.LSTM(input_size=128 * 3, hidden_size=256, num_layers=3, batch_first=True, bidirectional=True, dropout=0.4)
        
        self.root_head = nn.Sequential(
            nn.Linear(256 * 2, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_roots)
        )
        self.qual_head = nn.Sequential(
            nn.Linear(256 * 2, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_quals)
        )

    def forward(self, x):
        x = self.cnn(x)
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, -1)
        rnn_out, _ = self.rnn(x)
        return self.root_head(rnn_out), self.qual_head(rnn_out)
