"""
Sample predictive model.
The ingestion program will call `predict` to get a prediction for each test image and then save the predictions for scoring. The following two methods are required:
- predict: uses the model to perform predictions.
- load: reloads the model.
"""

import torch
import os
from transformers import AutoModel, AutoImageProcessor
import torch.nn as nn


def get_DINO():
    """function that returns frozen DINO model

    model: bioclip
    """
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    model = model.cuda()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    return model, processor


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


class Model:
    def __init__(self):
        # model will be called from the load() method
        self.model = None
        self.transforms = None

    def load(self):
        dino, processor = get_DINO()
        self.processor = processor
        model_path = os.path.join(os.path.dirname(__file__), "model.pth")
        self.model = DINO_DeepRegressor(dino=dino).cuda()
        self.model.eval()
        self.model.regressor.load_state_dict(torch.load(model_path))

    def predict(self, datapoints):
        images = [entry["relative_img"] for entry in datapoints]
        tensor_images = torch.stack(
            [
                self.processor(image, return_tensors="pt")["pixel_values"][0]
                for image in images
            ]
        )
        # model outputs 30d,1y,2y
        outputs = []
        dset = torch.utils.data.TensorDataset(tensor_images)
        loader = torch.utils.data.DataLoader(dset, batch_size=4, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                x = batch[0]
                out = self.model(x.cuda()).detach().cpu()
                if len(out.shape) == 1:
                    out = out.unsqueeze(0)
                outputs.append(out)
        outputs = torch.cat(outputs)
        mu = torch.mean(outputs, dim=0)
        sigma = torch.std(outputs, dim=0)
        return {
            "SPEI_30d": {"mu": mu[0].item(), "sigma": sigma[0].item()},
            "SPEI_1y": {"mu": mu[1].item(), "sigma": sigma[1].item()},
            "SPEI_2y": {"mu": mu[2].item(), "sigma": sigma[2].item()},
        }
