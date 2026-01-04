import torch.nn as nn


class DINO_DeepRegressor(nn.Module):
    def __init__(
        self, dino, hidden_size_begin=512, hidden_layer_decrease_factor=4, num_outputs=3
    ):
        super().__init__()
        # regressor linear layer
        self.dino = dino
        self.tokens_to_linear = nn.Sequential(
            nn.Conv2d(
                in_channels=768, out_channels=768, kernel_size=5, padding=0, stride=1
            ),  # Bx768x16x16 -> Bx768x12x12
            nn.ReLU(),
            nn.Conv2d(
                in_channels=768, out_channels=1024, kernel_size=12, stride=1, padding=0
            ),  # -> Bx1024x1x1
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(in_features=1024, out_features=hidden_size_begin),
            nn.GELU(),
            nn.Linear(
                in_features=hidden_size_begin,
                out_features=int(hidden_size_begin / hidden_layer_decrease_factor),
            ),
            nn.GELU(),
            nn.Linear(
                in_features=int(hidden_size_begin / hidden_layer_decrease_factor),
                out_features=int(hidden_size_begin / hidden_layer_decrease_factor**2),
            ),
            nn.GELU(),
            nn.Linear(
                in_features=int(hidden_size_begin / hidden_layer_decrease_factor**2),
                out_features=num_outputs,
            ),
        )

    def forward(self, x):
        features = self.dino(x)[0][:, 1:]
        # adjust features so Bx256x768 -> B x 768 x 16x16 so that positional data can be conserved(hopefully)?
        transposed_patches = features.transpose(1, 2)  # -> Bx768x256
        unflat = transposed_patches.unflatten(dim=2, sizes=(16, 16))  # -> B x768x16x16

        return self.regressor(self.tokens_to_linear(unflat).squeeze())
