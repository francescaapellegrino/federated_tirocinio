import flwr as fl
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler

# Funzione per caricare i dati dal CSV corrispondente al client_id

def load_data(client_id):
    file_path = f"data/binaryAllNaturalPlusNormalVsAttacks/data{client_id}.csv"
    df = pd.read_csv(file_path)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    X = df.drop(columns=["marker"])
    y = (df["marker"] != "Natural").astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values



# Classe client Flower
class SmartGridClient(fl.client.NumPyClient):
    def __init__(self, model, client_id):
        self.model = model
        self.client_id = client_id
        self.X_train, self.y_train = load_data(client_id)

    def get_parameters(self):
        # Estrai i pesi dal modello sklearn
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters):
        # Imposta i pesi nel modello sklearn
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.X_train, self.y_train)
        return self.get_parameters(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        preds = self.model.predict(self.X_train)
        loss = 1 - accuracy_score(self.y_train, preds)  # semplice loss come 1 - accuracy
        accuracy = accuracy_score(self.y_train, preds)
        return float(loss), len(self.X_train), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python client_sg.py <client_id>")
        sys.exit(1)

    client_id = int(sys.argv[1])
    model = LogisticRegression(max_iter=1000)
    client = SmartGridClient(model, client_id)

    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
