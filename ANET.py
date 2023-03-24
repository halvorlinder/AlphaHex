import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import numpy as np
import math
import torch.utils.data as data_utils
from functools import reduce
import wandb

from neural_net import NeuralNet
from representations import StateRepresentation

import CONSTANTS

DEBUG_MODE = False

class PytorchNN(NeuralNet):
    def __init__(self, model : nn.Module = None) -> None:
       self.model = model
       self.num_epochs = CONSTANTS.NUM_EPOCHS
       self.learning_rate = CONSTANTS.LEARNING_RATE
       self.batch_size = CONSTANTS.BATCH_SIZE

    def train(self, x : np.ndarray, y):
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        examples = data_utils.TensorDataset(x, y)
        trainer = Trainer(model=self.model, num_epochs=self.num_epochs, learning_rate=self.learning_rate, batch_size=self.batch_size)
        return trainer.train(examples)

    def predict(self, data : np.ndarray, device = CONSTANTS.DEVICE) -> np.ndarray:
        self.model.eval()
        output : torch.Tensor = self.model.forward(torch.tensor(np.array([data], dtype=float), dtype=torch.float32, device=device))
        return torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy().flatten()

    def save(self, filename: str):
        torch.save(self.model.state_dict(), filename)

    def load(self, model: nn.Module, filename: str):
        self.model = model
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()
    

class ConvNet(nn.Module):


    def __init__(self, board_state_length: int, board_dimension_depth: int, move_cardinality: int) -> None:
        super().__init__()
        self.state_representation = StateRepresentation.LAYERED
        self.max = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv1 = nn.Conv2d(in_channels=board_dimension_depth, out_channels=20, kernel_size=3, stride=1, padding="same")
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=3, stride=1, padding="same")
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3, stride=1, padding="same")
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=100, kernel_size=1, stride=1)
        self.fc1 = nn.Linear((math.isqrt(board_state_length)- 3) ** 2 * 100, 100)
        self.fc2 = nn.Linear(100, 40)
        self.fc3 = nn.Linear(40, move_cardinality)

    def forward(self, x):
        x = self.max(F.relu(self.conv1(x)))
        x = self.max(F.relu(self.conv2(x)))
        x = self.max(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x
    
class ConvResNet(nn.Module):

    def __init__(self, board_dimension_depth, channels, num_res_blocks, board_state_length, move_cardinality, device = CONSTANTS.DEVICE) -> None:
        super().__init__()

        self.device = device
        self.state_representation = StateRepresentation.LAYERED
        self.startBlock = nn.Sequential(
            nn.Conv2d(in_channels=board_dimension_depth, out_channels=channels, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(channels), 
            nn.ReLU()
        )
        self.resLayers = nn.ModuleList(
            [ResBlock(channels=channels) for _ in range(num_res_blocks)]
        )
        self.policyHead = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.Flatten(), 
            nn.Linear(in_features=32*board_state_length, out_features=move_cardinality)
        )

        self.to(device=device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.resLayers:
            x = resBlock(x)
        policy = self.policyHead(x)
        return policy

class ResBlock(nn.Module):

    def __init__(self, channels, stride = 1) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
                                nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1),
                                nn.BatchNorm2d(channels),
                                nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(channels))
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class FFNet(nn.Module):

    def __init__(self, board_state_length: int = None, move_cardinality: int = None, filename : str = None, device=CONSTANTS.DEVICE) -> None:
        super().__init__()
        self.state_representation = StateRepresentation.FLAT
        if not filename:
            self.layers = nn.ModuleList([nn.Sequential(
                nn.Linear(inp, out), 
                nn.BatchNorm1d(out), 
                nn.ReLU(), 
                nn.Dropout(CONSTANTS.DROPOUT_RATE)
                ) for (inp, out) in zip([board_state_length]+CONSTANTS.LAYERS[:-1], CONSTANTS.LAYERS[1:])])
            self.final_layer = nn.Sequential(
                nn.Linear(CONSTANTS.LAYERS[-1], move_cardinality)
            )
        else: 
            self.load_state_dict(torch.load(filename), map_location=CONSTANTS.DEVICE)

        self.to(device=CONSTANTS.DEVICE)

    def forward(self, x):
        x = reduce(lambda x, f: F.relu(f(x)), self.layers, x )
        x = self.final_layer(x)
        return x
    

class Trainer():

    def __init__(
        self, 
        model, 
        num_epochs, 
        learning_rate, 
        batch_size, 
        ) -> None:
        self.model = model
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = nn.MSELoss()

    def train_one_epoch(self, optimizer: torch.optim.Optimizer, training_loader, epoch: int):
        running_loss = 0.
        total_loss = 0.

        for i, data in enumerate(training_loader):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = self.model(inputs.to(CONSTANTS.DEVICE))

            if DEBUG_MODE and i == 0:
                print("INPUT")
                print(inputs)

                print("OUTPUTS")
                print(outputs)

                print("LABELS")
                print(labels)

            loss = self.loss_fn(outputs.to(CONSTANTS.DEVICE), labels.to(CONSTANTS.DEVICE))
            loss.backward()

            optimizer.step()
            if DEBUG_MODE:
                print(f"Loss: {loss}")

            running_loss += loss.item()
            total_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        return total_loss / (i+1)

    def train(self, tensor_dataset: data_utils.TensorDataset):
        #torch_dataset = torch.TensorDataset(training_set_tensor)
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        training_loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        total_loss_over_epochs = 0
        for epoch in range(self.num_epochs):
            # print(f"Training epoch {epoch+1}")
            self.model.train(True)
            avg_loss = self.train_one_epoch(optimizer=optimizer, training_loader=training_loader, epoch=epoch)
            total_loss_over_epochs += avg_loss
            # print("Loss: ", avg_loss)
        return total_loss_over_epochs / self.num_epochs



if __name__ == "__main__":
    model1 = ConvResNet(3, 32, 10, 16, 16)
    model2 = FFNet(16, 16)
    from torchsummary import summary
    summary(model1, (3, 4, 4))
    summary(model2, (16,))
    