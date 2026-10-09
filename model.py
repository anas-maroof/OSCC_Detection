import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Classifier(nn.Module):

    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()

        # Load pretrained ResNet-50
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        # ---------------------------------------------------
        # Freeze all layers
        # ---------------------------------------------------

        for param in self.backbone.parameters():
            param.requires_grad = False

        # ---------------------------------------------------
        # Unfreeze only Layer4
        # ---------------------------------------------------

        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        # ---------------------------------------------------
        # Replace classifier
        # ---------------------------------------------------

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(

            nn.Linear(in_features, 512),

            nn.BatchNorm1d(512),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(512, 128),

            nn.ReLU(inplace=True),

            nn.Dropout(0.4),

            nn.Linear(128, num_classes)

        )

        # Train classifier
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)
    
