
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import time
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define a simple PyTorch model
class Net(nn.Module):
    def __init__(self, input_size=784, output_size=10):
        super(Net, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

# Define a custom Flower client with power tracking
class FLClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.model = Net()
        self.cid = cid  # Client ID
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.power_consumption = 0  # Store power consumption per client

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict({k: torch.tensor(v) for k, v in state_dict.items()})

    def fit(self, parameters, config):
        start_time = time.time()  # Start time tracking

        self.set_parameters(parameters)
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Simulating dummy training data
        X_train = torch.randn(100, 784)
        y_train = torch.randint(0, 10, (100,))

        self.model.train()
        for epoch in range(1):  # 1 epoch for simulation
            optimizer.zero_grad()
            output = self.model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

        end_time = time.time()  # End time tracking

        # Calculate power consumption based on processing time and workload
        processing_time = end_time - start_time
        power_used = (5 + 0.1 * len(X_train)) * processing_time  # Example formula

        self.power_consumption = power_used
        logging.info(f"Client {self.cid}: Power Consumption = {self.power_consumption:.2f} W")

        # Send power consumption as a metric to the server
        return self.get_parameters(config), len(X_train), {"power_consumption": self.power_consumption}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        # Simulating evaluation
        X_test = torch.randn(20, 784)
        y_test = torch.randint(0, 10, (20,))

        self.model.eval()
        with torch.no_grad():
            output = self.model(X_test)
            loss = nn.CrossEntropyLoss()(output, y_test)
            accuracy = (output.argmax(dim=1) == y_test).float().mean().item()

        return float(loss), len(X_test), {"accuracy": accuracy}

# Start a single client process
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python fl_client.py <client_id>")
        sys.exit(1)

    client_id = int(sys.argv[1])
    client = FLClient(cid=client_id)
    
    # Connect to server
    fl.client.start_numpy_client(server_address="127.0.0.1:9091", client=client)

