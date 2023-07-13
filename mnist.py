import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from tqdm import trange
import os 
from safetensors import safe_open
from safetensors.torch import save_file

trainset = torchvision.datasets.MNIST(root="mnist_data", train=True, 
                                      transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(trainset, batch_size=16, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "mnist_classifier.safetensors"

class MNISTClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.L1 = nn.Linear(784, 128, bias=False)
        self.L2 = nn.Linear(128, 10, bias=False)
    
    def forward(self, x):
        x = self.L1(x)
        x = F.relu(x)
        x = self.L2(x)
        return x 

model = MNISTClassifier().to(device)

loss_function = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters())
epochs = 3

if __name__ == "__main__":
    if os.path.exists(model_path):
        print("model already exists")
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                model.state_dict()[key] = f.get_tensor(key)
                print(model.state_dict()[key])
            exit()

    for _ in (t := trange(epochs)):
        for x, y in dataloader:
            x,y = x.to(device).view(-1, 784), y.to(device)
            optim.zero_grad()
            output = model(x)
            loss = loss_function(output, y)
            loss.backward()
            optim.step()
            n_correct = torch.argmax(output, dim=1)
            acc = (n_correct == y).float().mean()
        t.set_description(f"acc = {acc.item():.2f} loss = {loss.item():.2f}")
    
    save_file(model.state_dict(), "mnist_classifier.safetensors")

        