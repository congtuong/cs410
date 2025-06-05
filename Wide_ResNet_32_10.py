import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual unit with 2 sub layers."""

    def __init__(self, in_filter, out_filter, stride=1, activate_before_residual=False):
        super(ResidualBlock, self).__init__()
        self.activate_before_residual = activate_before_residual
        self.in_filter = in_filter
        self.out_filter = out_filter
        self.stride = stride

        # First batch norm and activation
        self.bn1 = nn.BatchNorm2d(in_filter)

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_filter, out_filter, kernel_size=3, stride=stride, padding=1, bias=False
        )

        # Second batch norm and activation
        self.bn2 = nn.BatchNorm2d(out_filter)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            out_filter, out_filter, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x):
        if self.activate_before_residual:
            out = F.leaky_relu(self.bn1(x), negative_slope=0.1)
            identity = out
        else:
            identity = x
            out = F.leaky_relu(self.bn1(x), negative_slope=0.1)

        out = self.conv1(out)
        out = self.conv2(F.leaky_relu(self.bn2(out), negative_slope=0.1))

        # Handle identity mapping with dimension change
        if self.in_filter != self.out_filter:
            identity = F.avg_pool2d(identity, self.stride)
            # Pad the channels with zeros
            pad_size = (self.out_filter - self.in_filter) // 2
            if pad_size > 0:
                identity = F.pad(identity, [0, 0, 0, 0, pad_size, pad_size])

        out += identity
        return out


class Model(nn.Module):
    """ResNet model."""

    def __init__(self, mode="eval", dataset="cifar10", train_batch_size=None):
        """ResNet constructor.

        Args:
          mode: One of 'train' and 'eval'.
          dataset: Dataset to use ('cifar10' or 'cifar100')
          train_batch_size: Batch size for training
        """
        super(Model, self).__init__()

        self.mode = mode
        self.num_classes = 100 if dataset == "cifar100" else 10
        self.train_batch_size = train_batch_size

        # Initial convolution
        self.init_conv = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Wide ResNet 28-10 configuration
        filters = [16, 160, 320, 640]
        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]

        # Create the residual blocks
        self.layers = nn.ModuleList()

        # Block 1
        self.layers.append(
            ResidualBlock(
                filters[0], filters[1], strides[0], activate_before_residual[0]
            )
        )
        for i in range(1, 5):
            self.layers.append(ResidualBlock(filters[1], filters[1], 1, False))

        # Block 2
        self.layers.append(
            ResidualBlock(
                filters[1], filters[2], strides[1], activate_before_residual[1]
            )
        )
        for i in range(1, 5):
            self.layers.append(ResidualBlock(filters[2], filters[2], 1, False))

        # Block 3
        self.layers.append(
            ResidualBlock(
                filters[2], filters[3], strides[2], activate_before_residual[2]
            )
        )
        for i in range(1, 5):
            self.layers.append(ResidualBlock(filters[3], filters[3], 1, False))

        # Final batch norm and activation
        self.final_bn = nn.BatchNorm2d(filters[3])

        # Fully connected layer
        self.fc = nn.Linear(filters[3], self.num_classes)

        # Initialize perturbation if in train mode
        if self.mode == "train" and train_batch_size is not None:
            self.register_parameter(
                "pert", nn.Parameter(torch.zeros(train_batch_size, 3, 32, 32))
            )
        else:
            self.pert = None

    def forward(self, x):
        # Apply perturbation if in training mode
        if self.mode == "train" and self.pert is not None:
            x = x + self.pert
            x = torch.clamp(x, 0.0, 1)

        # Per-image standardization
        # Note: This is different from PyTorch's standard normalization
        # We compute mean and std for each image separately
        batch_size = x.size(0)
        mean = torch.mean(x.view(batch_size, -1), dim=1).view(batch_size, 1, 1, 1)
        std = torch.std(x.view(batch_size, -1), dim=1).view(batch_size, 1, 1, 1)
        x = (x - mean) / (std + 1e-5)

        # Initial convolution
        x = self.init_conv(x)

        # Apply residual blocks
        for layer in self.layers:
            x = layer(x)

        # Final batch norm and activation
        x = F.leaky_relu(self.final_bn(x), negative_slope=0.1)

        # Global average pooling
        x = F.avg_pool2d(x, x.size(2))
        neck = x.view(x.size(0), -1)

        # Fully connected layer
        logits = self.fc(neck)

        return logits, neck

    def _weight_decay(self):
        """L2 weight decay loss."""
        decay_loss = 0.0
        for name, param in self.named_parameters():
            if "weight" in name:  # Only apply to weights, not biases
                decay_loss += torch.sum(torch.square(param))
        return decay_loss / 2.0  # Divide by 2 to match TensorFlow's implementation
