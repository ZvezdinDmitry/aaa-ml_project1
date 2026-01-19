from torch import nn


class EmbeddingMLP(nn.Module):
    def __init__(self, dim):
        super(EmbeddingMLP, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.1)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1)
        )

        self.mlp3 = nn.Sequential(nn.Linear(128, 128), nn.ReLU())

        self.predict_logit = nn.Linear(128, 4)

    def forward(self, x):
        x1 = self.mlp1(x)
        x2 = self.mlp2(x1)
        x3 = self.mlp3(x2 + x1)
        return self.predict_logit(x3)
