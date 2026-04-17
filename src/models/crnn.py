import torch
import torch.nn as nn

class ChordCRNN(nn.Module):
    """
    Baseline architecture: Simple CNN + Bi-LSTM.
    """
    def __init__(self, num_classes=25):
        super(ChordCRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        # 64 channels * 3 frequency bins = 192
        self.rnn = nn.LSTM(64 * 3, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x) 
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, -1) 
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x
