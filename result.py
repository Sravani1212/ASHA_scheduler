import torch
import torch.nn as nn
#from ray.tune.analysis import Analysis
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def get_dataset():
    return datasets.CIFAR10(
        root='./data', train=True, download=True, transform=ToTensor(),
    )

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32 * 32 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, inputs):
        inputs = self.flatten(inputs)
        logits = self.linear_relu_stack(inputs)
        return logits

def train_func(config):
    num_epochs = 3
    batch_size = 64

    dataset = get_dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = NeuralNetwork()

    criterion = nn.CrossEntropyLoss()
    
    # Retrieve the learning rate and momentum from the config
    lr = config["lr"]
    momentum = config["momentum"]

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            pred = model(inputs)
            accuracy = criterion(pred, labels)
            accuracy.backward()
            optimizer.step()
        print(f"epoch: {epoch}, accuracy: {accuracy.item()}")
    
    return {"accuracy": accuracy.item()}

# For GPU Training, set `use_gpu` to True.
use_gpu = True

asha_scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric='accuracy',
    mode='max',
    max_t=100,
    grace_period=10,
    reduction_factor=3,
    brackets=1,
)

tuner = tune.run(
    train_func,
    config={
        "lr": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
        "momentum": tune.grid_search([0.9, 0.95, 0.99]),
    },
    num_samples=10,
    scheduler=asha_scheduler
)
print(tuner.get_best_trial(metric="accuracy", mode="max").config)