import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.estimators.classification import PyTorchClassifier
import numpy as np

# --- Definizione modello PyTorch semplice ---
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# --- Setup dati ---
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Client Flower con attacco MIA ---
class FlowerClientWithMIA(fl.client.NumPyClient):
    def __init__(self):
        self.model = Net().to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Wrappiamo il modello PyTorch per ART
        self.classifier = PyTorchClassifier(
            model=self.model,
            loss=self.loss_fn,
            optimizer=self.optimizer,
            input_shape=(1, 28, 28),
            nb_classes=10,
            device_type='gpu' if torch.cuda.is_available() else 'cpu'
        )

    def get_parameters(self):
        # Ritorna i parametri come numpy array
        return [val.cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        # Setta i parametri nel modello
        params_torch = [torch.tensor(p).to(device) for p in parameters]
        for p, p_new in zip(self.model.parameters(), params_torch):
            p.data = p_new.data.clone()

    def train(self, epochs=1):
        self.model.train()
        for _ in range(epochs):
            for x_batch, y_batch in trainloader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                self.optimizer.zero_grad()
                output = self.model(x_batch)
                loss = self.loss_fn(output, y_batch)
                loss.backward()
                self.optimizer.step()

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in testloader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = self.model(x_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        return correct / total

    def generate_attack_data(self):
        # Prepara dati per attacco MIA (prendiamo un subset dei dati)
        x_attack = []
        y_attack = []

        # Qui scegliamo una porzione di dati di train (membri) e test (non membri)
        for i, (x, y) in enumerate(trainloader):
            if i > 2:
                break
            x_attack.append(x.numpy())
            y_attack.append(y.numpy())
        for i, (x, y) in enumerate(testloader):
            if i > 2:
                break
            x_attack.append(x.numpy())
            y_attack.append(y.numpy())

        x_attack = np.vstack(x_attack)
        y_attack = np.hstack(y_attack)
        return x_attack, y_attack

    def perform_mia(self):
        # Dati per l'attacco
        x_attack, y_attack = self.generate_attack_data()

        # Crea l'attacco MIA BlackBox (serve il modello wrapped in ART)
        mia = MembershipInferenceBlackBox(classifier=self.classifier)

        # Esegui l'attacco (ottiene array di predizioni membership)
        membership_preds = mia.infer(x_attack, y_attack)

        # membership_preds è un array booleano con predizione di membership
        membership_accuracy = np.mean(membership_preds == np.array([True]*(len(x_attack)//2) + [False]*(len(x_attack)//2)))
        print(f"[MIA] Membership inference accuracy: {membership_accuracy:.4f}")

    # --- Metodi Flower ---
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train(epochs=1)
        return self.get_parameters(), len(trainset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        accuracy = self.evaluate()
        print(f"Valutazione locale: accuracy={accuracy:.4f}")

        # Esegui attacco di inferenza dopo valutazione
        self.perform_mia()

        return float(1 - accuracy), len(testset), {"accuracy": accuracy}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClientWithMIA())