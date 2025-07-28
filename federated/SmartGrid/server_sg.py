import flwr as fl
from sklearn.linear_model import LogisticRegression
import numpy as np

def get_initial_parameters():
    model = LogisticRegression(max_iter=1000)
    # Inizializza parametri a zero per compatibilità
    coef = np.zeros((1, 110))  # modifica 110 con il numero esatto di feature
    intercept = np.zeros(1)
    return [coef, intercept]

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    initial_parameters=fl.common.ndarrays_to_parameters(get_initial_parameters())
)

class Config:
    def __init__(self):
        self.num_rounds = 3
        self.round_timeout = None 

if __name__ == "__main__":
    fl.server.start_server(
        server_address="localhost:8080",
        config=Config(),
        strategy=strategy,
    )
