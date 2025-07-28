'''
Crea una rete neurale fully conneted da zero,
la addestra su un dataset reale (MNIST)
valuta le prestazioni con metriche standard
'''

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# I tensor verranno elaborati sulla GPU CUDA, se c'è, altrimenti sulla CPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset MNIST:

# Immagini trasformate in tensor e le normalizza 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# Scarica la parte di addestramento del dataset MNIST se non è già presente
trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
# Suddivide i dati in mini slot e li mescola
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
# Lo stesso per la parte di test
testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# Rete neurale semplice
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Funzione di addestramento
def train(model, loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(loader):.4f}")

# Funzione di valutazione
def test(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    print(f"Test Loss: {loss / len(loader):.4f}")
    print(f"Accuracy: {correct / total * 100:.2f}%")

# MAIN
if __name__ == "__main__":
    net = Net().to(DEVICE)
    train(net, trainloader, epochs=5)
    test(net, testloader)