import lightning.pytorch as pl
from transformers import MobileViTForImageClassification
from efficientnet_pytorch import EfficientNet
import torch
from torch import nn

class GenericModelWrapper(pl.LightningModule):
    def __init__(self, model, feature_extractor, output_size):
        super(GenericModelWrapper, self).__init__()
        self.model = model
        self.feature_extractor = feature_extractor
        self.output_size = output_size

    def forward(self, x, return_features=False):
        if return_features:
            return self.feature_extractor(x)
        else:
            return self.model(x).logits

    def get_feature_map(self, x):
        with torch.no_grad():
            self.eval()
            features = self(x, return_features=True)
            return features

class MobileViTWrapper(GenericModelWrapper):
    def __init__(self, model_ckpt, num_labels):
        model = MobileViTForImageClassification.from_pretrained(
            model_ckpt, num_labels=num_labels, ignore_mismatched_sizes=True
        )
        super(MobileViTWrapper, self).__init__(model, self.extract_features, num_labels)

    def extract_features(self, x):
        outputs = self.model(x, output_hidden_states=True)
        return outputs.hidden_states[-1]

class EfficientNetWrapper(GenericModelWrapper):
    def __init__(self, model_name, num_classes):
        model = EfficientNet.from_pretrained(model_name)
        output_size = model._fc.in_features
        model._fc = nn.Linear(output_size, num_classes)
        super(EfficientNetWrapper, self).__init__(model, self.extract_features, output_size)

    def extract_features(self, x):
        return self.model.extract_features(x)
