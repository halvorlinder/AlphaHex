import torch
import torch.nn as nn
import torch.nn.functional as F

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
    

class Trainer():

    def __init__(
        self, 
        model, 
        num_epochs, 
        learning_rate, 
        batch_size, 
        test_size = 0.2, 
        ) -> None:
        self.model = model
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.test_size = test_size
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, training_set):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        training_loader = torch.utils.data.DataLoader(training_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

        for epoch in range(self.num_epochs):
            print(f"Training epoch {epoch+1}")

            for i, data in enumerate(training_loader):
                inputs, labels = data

                optimizer.zero_grad()

                outputs = self.model(inputs)

                loss = 
