from model_wrappers import MobileViTWrapper, EfficientNetWrapper
from ensemble_model import SimpleEnsembleModel, AdvancedEnsembleModel

def create_ensemble_model(model_wrappers, num_classes, use_advanced=False):
    if use_advanced:
        return AdvancedEnsembleModel(model_wrappers, num_classes)
    else:
        return SimpleEnsembleModel(model_wrappers, num_classes)
