from .model_wrappers import ModelWrapper
from .ensemble_model import SimpleEnsembleModel, AdvancedEnsembleModel

def create_ensemble_model(model_wrappers, num_classes, use_advanced=False):
    wrapped_models = [ModelWrapper(model) for model in model_wrappers]
    if use_advanced:
        return AdvancedEnsembleModel(wrapped_models, num_classes)
    else:
        return SimpleEnsembleModel(wrapped_models, num_classes)
