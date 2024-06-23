import torch
from torch import nn
from torchviz import make_dot
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from models import SimpleEnsembleModel, MediumEnsembleModel, AdvancedEnsembleModel

def visualize_model(model, input_size):
    # Create a batch with more than one sample
    x = torch.randn(2, *input_size)
    y = model(x)
    graph = make_dot(y, params=dict(model.named_parameters()))
    return graph

feature_size = 160
num_classes = 4
input_size = (feature_size,)
input_size_adv = (10, feature_size)

simple_model = SimpleEnsembleModel(feature_size, num_classes)
medium_model = MediumEnsembleModel(feature_size, num_classes)
advanced_model = AdvancedEnsembleModel(feature_size, num_classes)

simple_graph = visualize_model(simple_model, input_size)
simple_graph.render("plots/SimpleEnsembleModel", format="pdf")

medium_graph = visualize_model(medium_model, input_size)
medium_graph.render("plots/MediumEnsembleModel", format="pdf")

advanced_graph = visualize_model(advanced_model, input_size_adv)
advanced_graph.render("plots/AdvancedEnsembleModel", format="pdf")