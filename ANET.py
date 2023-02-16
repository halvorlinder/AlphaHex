import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import numpy as np
import math

from neural_net import NeuralNet
from representations import StateRepresentation

DEBUG_MODE = True

class PytorchNN(NeuralNet):
    def __init__(self, model : nn.Module = None, num_epochs : int = 1, learning_rate : float = 0.01, batch_size : int = 4) -> None:
       self.model = model
       self.num_epochs = num_epochs
       self.learning_rate = learning_rate
       self.batch_size = batch_size

    def train(self, examples : np.ndarray):
        trainer = Trainer(model=self.model, num_epochs=self.num_epochs, learning_rate=self.learning_rate, batch_size=self.batch_size)
        print(np.array(examples))
        trainer.train(np.array(examples))

    def predict(self, data : np.ndarray) -> np.ndarray:
        # print(data)
        # print(np.array([data], dtype=float))
        output : torch.Tensor = self.model.forward(torch.tensor(np.array([data], dtype=float)).to(torch.float32))
        return output.detach().numpy().flatten()

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
            self.fc1 = nn.Linear(board_state_length, 40)
            self.fc2 = nn.Linear(40, 100)
            self.fc3 = nn.Linear(100, 40)
            self.fc4 = nn.Linear(40, move_cardinality)
        else: 
            self.load_state_dict(torch.load(filename))

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
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

    def train_one_epoch(self, optimizer: torch.optim.Optimizer, training_loader, epoch: int):
        running_loss = 0.

        for i, data in enumerate(training_loader):
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
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        return running_loss / (i+1)

    def train(self, training_set: np.array):
        # TODO: this should be an array of shape (Cbatch, channels, height, width or ()C, H, W
        print(training_set.shape)
        training_set_tensor = torch.tensor(training_set, dtype=torch.float32)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        training_loader = torch.utils.data.DataLoader(training_set_tensor, batch_size=self.batch_size, shuffle=True, num_workers=0)

        for epoch in range(self.num_epochs):
            print(f"Training epoch {epoch+1}")

            self.model.train(True)

            avg_loss = self.train_one_epoch(optimizer=optimizer, training_loader=training_loader, epoch=epoch)
            
            # print("Loss: ", avg_loss)

            #model_path = 'model_{}_{}'.format("fake", epoch)
            #self.model.save(model_path)



if __name__ == "__main__":
    model = ConvNet(25, 3, 25)
    from torchsummary import summary
    summary(model, (3, 5, 5))

    
