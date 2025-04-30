# 🚀 Simulation of Edge Computing using Federated Learning

This project demonstrates a simulation of Federated Learning (FL) applied to edge computing using Flower (FLwr). It simulates multiple edge servers (clients) training locally on partitioned datasets and sending model updates to a central FL server. The primary goal is to analyze and compare energy consumption of edge servers during training against existing baselines such as those used in EdgeAISim.

---

## 📌 Project Objectives

- Simulate real-world edge computing with FL across multiple edge devices.
- Measure power consumption for each edge server per FL training round.
- Compare simulation results against EdgeAISim baselines like Q-Learning, MAB-UCB, Actor-Critic, and Worst Fit in terms of energy usage.
- Visualize energy usage comparisons effectively.

---

## 📂 Project Structure

federated_learning/ │ ├── fl_server.py # Flower-based Federated Learning server ├── fl_client.py # Flower client running on simulated edge servers ├── fl_model.py # Model architecture (e.g., simple CNN or MLP) ├── fl_dataset.py # Dataset loading and client-specific partitioning ├── fl_utils.py # Utilities: power consumption tracking, metrics ├── fl_start.py # Script to launch server and clients ├── fl_config.py # Configuration file for model, rounds, dataset ├── requirements.txt # Required Python libraries ├── README.md # Project documentation (this file)



---

## ⚙️ How to Run

1. **Clone the Repository**

git clone https://github.com/yourusername/edge-fl-simulator.git
cd edge-fl-simulator



📊 Output
Prints power consumption of each edge server per round.

Saves a comparison of power metrics to CSV/console.

Visualizes consumption vs EdgeAISim baseline methods.



🔋 Sample Output Table

| Server         | FL Power (W) | EdgeAISim GNN (W) |
|----------------|--------------|-------------------|
| Edge Server 1  | 170.5        | 181.4             |
| Edge Server 2  | 160.2        | 166.24            |
| Edge Server 3  | 184.1        | 184.5             |
| Edge Server 4  | 198.7        | 200               |
| Edge Server 5  | 80.5         | 81                |
| Edge Server 6  | 63.1         | 63                |
| **Total**      | **920.1**    | **1000**          |



📈 Comparison with EdgeAISim

| Works Models         | Edge Server 1 | Edge Server 2 | Edge Server 3 | Edge Server 4 | Edge Server 5 | Edge Server 6 | Total Power |
|----------------------|---------------|---------------|----------------|----------------|----------------|----------------|--------------|
| EdgeAISim-GNN        | 181.4         | 166.24        | 184.5          | 200            | 81             | 63             | 1000         |
| Q-Learning           | 307           | 180           | 85.9           | 91.25          | 90.5           | 82.4           | 4875         |
| Actor-Critic         | 236.5         | 188.5         | 94.4           | 96             | 81.9           | 79.6           | 12850        |
| MAB-UCB              | 217.5         | 227           | 163            | 161.5          | 128.6          | 129            | 5600         |
| Worst Fit (Baseline) | 220           | 192.57        | 77.85          | 80.85          | 118.5          | 125            | 5431         |
| **This Work (FL)**   | 170.5         | 160.2         | 184.1          | 198.7          | 80.5           | 63.1           | **920.1**    |
