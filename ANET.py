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
        # print(np.array(examples))
        return trainer.train(examples)

    # def train(self, examples : np.ndarray):
    #     trainer = Trainer(model=self.model, num_epochs=self.num_epochs, learning_rate=self.learning_rate, batch_size=self.batch_size)
    #     trainer.train(examples)

    def predict(self, data : np.ndarray) -> np.ndarray:
        # print(data)
        # print(np.array([data], dtype=float))
        output : torch.Tensor = self.model.forward(torch.tensor(np.array([data], dtype=float)).to(torch.float32))
        return torch.nn.functional.softmax(output, dim=1).detach().numpy().flatten()

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




class FFNet(nn.Module):

    def __init__(self, board_state_length: int = None, move_cardinality: int = None, filename : str = None) -> None:
        super().__init__()
        self.state_representation = StateRepresentation.FLAT
        if not filename:
            self.layers = nn.ModuleList([nn.Linear(inp, out) for (inp, out) in zip([board_state_length]+CONSTANTS.LAYERS, CONSTANTS.LAYERS+[move_cardinality])])
        else: 
            self.load_state_dict(torch.load(filename))

    def forward(self, x):
        return reduce(lambda x, f: F.relu(f(x)), self.layers, x )
    

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
            # print(i)
            inputs, labels = data

            optimizer.zero_grad()

            outputs = self.model(inputs)

            if DEBUG_MODE and i == 0:
                print("INPUT")
                print(inputs)

                print("OUTPUTS")
                print(outputs)

                print("LABELS")
                print(labels)

            loss = self.loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()
            if DEBUG_MODE:
                print(f"Loss: {loss}")

            running_loss += loss.item()
            total_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        return total_loss / (i+1)

    def train(self, tensor_dataset: data_utils.TensorDataset):
        #torch_dataset = torch.TensorDataset(training_set_tensor)
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        training_loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        total_loss_over_epochs = 0
        for epoch in range(self.num_epochs):
            print(f"Training epoch {epoch+1}")

            self.model.train(True)

            avg_loss = self.train_one_epoch(optimizer=optimizer, training_loader=training_loader, epoch=epoch)
            total_loss_over_epochs += avg_loss
            print("Loss: ", avg_loss)


            #model_path = 'model_{}_{}'.format("fake", epoch)
            #self.model.save(model_path)
        return total_loss_over_epochs / self.num_epochs



if __name__ == "__main__":
    # model = ConvNet(25, 3, 25)
    # from torchsummary import summary
    # summary(model, (3, 5, 5))
    import matplotlib.pyplot as plt

    x = np.random.uniform(-5, 5, 1000)
    x = x.reshape(1000, 1)

    noise = np.random.uniform(-0.1, 0.1, 1000).reshape(1000, 1)
    y = np.add(np.power(x, 2), noise)

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    net = FFNet(1, 1)

    trainer = Trainer(net, 4, 0.001, 8)

    train_set = data_utils.TensorDataset(x, y)

    trainer.train(train_set)
    
    vals = net(x.clone().detach())
    vals = vals.detach().numpy().flatten()
    plt.plot(x, vals, "*", color="red")
    plt.plot(x, y, "*", color="blue")
    plt.show()
    