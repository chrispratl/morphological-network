import os

import torch
from torch import nn
from torch.utils.data import DataLoader

# Which data to use
# from data.mnist_data import training_data, test_data
from data.mnist_fashion_data import training_data, test_data

from morph_network import MorphologicalMax

# Where to store model data
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "fashionmnist.pt"
)

BATCH_SIZE = 64
EPOCHS = 300

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

train_dataLoader = DataLoader(training_data, batch_size=BATCH_SIZE)
test_dataLoader = DataLoader(test_data, batch_size=BATCH_SIZE)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            MorphologicalMax(28*28, 200, 200, 10),
            # MorphologicalMax(200, 200, 200, 10)
        )

    def forward(self, x):

        self.x_orig = x
        x = self.flatten(x)

        logits = self.stack(x)

        return logits

def train_epoch(dataloader, model, loss_fn, optimizer):
    """
        Do a training step with all the training data
    """

    size = len(dataloader.dataset)

    # Start training of the model
    model.train()

    for batch, (X, y) in enumerate(dataloader):

        # Store the data on the correct device
        X, y = X.to(device), y.to(device)

        # Forward step
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backward step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Just some informational data. Uncomment
        # if you are interested.
        # if batch % 100 == 99:
        #     loss, current = loss.item(), (batch + 1) * len(X)
        #     print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, epoch):
    """
        Check accuracy, based on the test data
    """
    size = len(dataloader.dataset)

    num_batches = len(dataloader)

    model.eval()

    test_loss, correct = 0, 0

    with torch.no_grad():

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)

            test_loss += loss_fn(pred, y).item()

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"Epoch {epoch + 1}: {(100*correct):>0.1f}% accurracy, avg. loss {test_loss:>8f}")

def train(epochs, model):
    """
        Do enough training epochs.
    """

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for t in range(epochs):
        # print(f"Epoch {t + 1}")
        # print("--------------")
        train_epoch(train_dataLoader, model, loss_fn, optimizer)
        test(test_dataLoader, model, loss_fn, t)

if __name__ == "__main__":
    model = NeuralNetwork().to(device)
    train(EPOCHS, model)
