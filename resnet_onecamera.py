import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import models
"""
* @brief Class declaration for building the training network model.
* @param None.
* @return The built netwwork.
"""
class model_cnn(nn.Module):
    """
    * @brief Initializes the class varaibles
    * @param None.
    * @return None.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=24):
        super().__init__()
        # resnet50 part for image processing
        model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=24), nn.Tanh())
        self.resnet = model
        # fc part for xy coordinates processing
        self.fc01 = nn.Linear(2, 24)
        # fc part for combining image part and xy part
        self.fc02
        # lstm part
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_img, input_xy):
        x1 = self.resnet(input_img[0])
        x2 = self.resnet(input_img[1])
        x3 = self.resnet(input_img[2])
        x4 = self.resnet(input_img[3])

        return input
