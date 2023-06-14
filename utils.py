import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def build_mnist(
    set: str, transforms: transforms.Compose, datadir: str = "./data", **kwargs
):
    ds = datasets.MNIST(
        datadir,
        train=True if set == "train" else False,
        download=True,
        transform=transforms,
    )
    data_loader = torch.utils.data.DataLoader(ds, **kwargs)

    return ds, data_loader


def plot_sampledata(loader: DataLoader):
    batch_data, batch_label = next(iter(loader))

    fig = plt.figure()

    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap="gray")
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

    plt.show()


def GetCorrectPredCount(pPrediction: torch, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


class Trainer:
    def __init__(self, model: nn.Module, device: torch.device, optimizer, scheduler = None) -> None:
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.train_acc = []
        self.train_losses = []
        self.test_acc = []
        self.test_losses = []

    def train(self, train_loader: DataLoader):
        self.model.train()
        pbar = tqdm(train_loader)

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # Predict
            pred = self.model(data)

            # Calculate loss
            loss = F.nll_loss(pred, target)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            correct += GetCorrectPredCount(pred, target)
            processed += len(data)

            pbar.set_description(
                desc=f"Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}"
            )

        self.train_acc.append(100 * correct / processed)
        self.train_losses.append(train_loss / len(train_loader))

    def test(self, test_loader: DataLoader):
        self.model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                test_loss += F.nll_loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss

                correct += GetCorrectPredCount(output, target)

        test_loss /= len(test_loader.dataset)
        self.test_acc.append(100.0 * correct / len(test_loader.dataset))
        self.test_losses.append(test_loss)

        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )


    def plot_history(self):
        max_train = max(self.train_acc)
        ep_train = self.train_acc.index(max_train) + 1
        
        max_test = max(self.test_acc)
        ep_test = self.test_acc.index(max_test) + 1
        print('Set\\t Max Acc@Epoch\t Last Epoch Acc')
        print(f'train\t {max_train:0.2f}@{ep_train}\t\t{self.train_acc[-1]:0.2f}')
        print(f'test\t {max_test:0.2f}@{ep_test}\t\t{self.test_acc[-1]:0.2f}')
        
        # For loss and epochs
        plt.figure(figsize=(14,5))
        plt.subplot(1,2,1)
        plt.plot(self.train_losses, label='Training Loss') # plotting the training loss
        plt.plot(self.test_losses, label='Test Loss') # plotting the testing loss
        # putting the labels on plot
        plt.title('Loss vs Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()

        # For accuracy and epochs
        plt.subplot(1,2,2)
        plt.plot(self.train_acc, label='Training Accuracy') # plotting the training accuracy
        plt.plot(self.test_acc, label='Test Accuracy') # plotting the testing accuracy
        # putting the labels in plot
        plt.title('Accuracy vs Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.legend()

        plt.show()


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device):
    cols, rows = 4,6
    figure = plt.figure(figsize=(20, 20))
    for i in range(1, cols * rows + 1):
        k = np.random.randint(0, len(loader.dataset)) # random points from test dataset
    
        img, label = loader.dataset[k] # separate the image and label
        img = img.unsqueeze(0) # adding one dimention
        pred=  model(img.to(device)) # Prediction 

        figure.add_subplot(rows, cols, i) # making the figure 
        plt.title(f"Predcited label {pred.argmax().item()}\n True Label: {label}") # title of plot
        plt.axis("off") # hiding the axis
        plt.imshow(img.squeeze(), cmap="gray") # showing the plot

    plt.show()