import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple PyTorch model for the server
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(784, 10)  # Example for MNIST dataset

    def forward(self, x):
        return self.fc(x)

# Initialize the global model
global_model = Net()
    
# Define a function to aggr
# egate power consumption
total_power_consumption = 0.0

def aggregate_fit(rnd, results, failures):
    global total_power_consumption

    print(f"🛠 Aggregating updates from {len(results)} clients in Round {rnd}...")

    # Extract client updates
    aggregated_weights = []
    for _, fit_res in results:
        aggregated_weights.append(fit_res.parameters)

        # Extract power consumption from metadata (Assuming clients send it)
        power_used = fit_res.metrics.get("power_consumption", 0.0)
        total_power_consumption += power_used

    # Aggregate model parameters (FedAvg)
    aggregated_parameters = fl.common.weights_to_parameters(aggregated_weights)
    
    print(f"📊 Total Power Consumption so far: {total_power_consumption:.2f} W")

    return aggregated_parameters, {}

# Define FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # All clients participate
    fraction_evaluate=1.0,
    min_fit_clients=2, 
    min_evaluate_clients=2,
    min_available_clients=2,
    on_fit_config_fn=lambda rnd: {"round": rnd},
    evaluate_metrics_aggregation_fn=aggregate_fit,
)

# Start Flower server
if __name__ == "__main__":
    print("🚀 Starting Federated Learning Server...")
    
    try:
        fl.server.start_server(server_address="127.0.0.1:9091", config=fl.server.ServerConfig(num_rounds=5), strategy=strategy)

        # Print final power consumption after training
        print(f"⚡ Final Total Power Consumption: {total_power_consumption:.2f} W")

    except Exception as e:
        print(f"❌ Server Error: {e}")
