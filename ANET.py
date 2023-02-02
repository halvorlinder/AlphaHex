import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import numpy as np

DEBUG_MODE = True

class FFNet(nn.Module):

    def __init__(self, board_state_length: int, move_cardinality: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(board_state_length, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, move_cardinality)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x))
        return x
    
    def save(self, filename: str):
        torch.save(self.state_dict(), filename)

    def load(self, filename: str):
        self.load_state_dict(torch.load(filename))
    

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

    def train_one_epoch(self, optimizer, training_loader):
        running_loss = 0.

        for i, data in enumerate(training_loader):
            inputs, labels = data

            if DEBUG_MODE and i == 0:
                print("INPUT")
                print(input)

                print("LABELS")
                print(labels)


            optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.loss_fn(outputs, labels)

            if DEBUG_MODE:
                print(f"Loss: {loss}")

            running_loss += loss

            loss.backward()

            optimizer.step()
        
        return running_loss / (i+1)

    def train(self, training_set: np.array):
        training_set_tensor = torch.tensor(training_set)
        torch_dataset = torch.TensorDataset(training_set_tensor)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        training_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

        for epoch in range(self.num_epochs):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            print(f"Training epoch {epoch+1}")

            self.model.train(True)

            avg_loss = self.train_one_epoch(optimizer=optimizer, training_loader=training_loader)
            
            print(avg_loss)

            model_path = 'model_{}_{}'.format(timestamp, epoch)
            self.model.save(model_path)
