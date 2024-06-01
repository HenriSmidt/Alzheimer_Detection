import torch
import torch.nn as nn
from .efficientnet_baseline import EfficientNetBaseline
from .mobilevit_lightning import MobileViTLightning

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def extract_features(self, x):
        return self.model.extract_features(x)

    @property
    def output_size(self):
        # Implement logic to return the correct output size
        # This might depend on the model architecture
        if isinstance(self.model, EfficientNetBaseline):
            return self.model.model._fc.in_features
        elif isinstance(self.model, MobileViTLightning):
            # Assuming 768 as the feature size for MobileViT
            return 768
        else:
            raise ValueError("Unsupported model type")
