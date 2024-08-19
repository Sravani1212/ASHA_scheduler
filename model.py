import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import matplotlib.pyplot as plt

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

    total_loss = 0.0

    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Return the average loss over all batches
    return {"loss": total_loss / len(dataloader)}

def plot_learning_curves(results):
    for trial in results.trials:
        plt.plot(trial.metric_analysis["loss"]["training_iteration"], trial.metric_analysis["loss"]["value"], label=str(trial.config))

    plt.title("Learning Curves for Different Hyperparameter Configurations")
    plt.xlabel("Training Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# For GPU Training, set `use_gpu` to True.
use_gpu = True

asha_scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric='loss',
    mode='min',
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

# Visualize learning curves
plot_learning_curves(tuner)

# Print terminated hyperparameter values
for trial in tuner.trials:
    if trial.status == "TERMINATED":
        print(f"Hyperparameters: {trial.config}, Loss: {trial.last_result['loss']}")
