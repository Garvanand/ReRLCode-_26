# Multi-Agent Deep RL for Smart Urban Traffic Signal Control

This project implements a production-grade Reinforcement Learning system to optimize traffic light timings in an urban grid using **SUMO (Simulation of Urban Mobility)** and **Proximal Policy Optimization (PPO)**.

## 🚀 Overview

Traffic congestion is a major urban problem. Conventional fixed-time traffic signals are inefficient under varying traffic loads. This project uses a Multi-Agent RL approach where each intersection is an independent agent, trained centrally using PPO to minimize waiting times and maximize throughput.

### Key Features
- **Algorithm:** PPO with Actor-Critic architecture.
- **Simulator:** SUMO (TraCI API).
- **Architecture:** Centralized Training, Decentralized Execution.
- **Environment:** 6×6 Grid with mixed vehicle types (cars, trucks, buses, motorcycles, delivery vans).
- **Metrics:** Waiting time, Queue length, Throughput, CO2 emissions proxy.

---

## 📂 Project Structure

```text
traffic_rl/
│
├── config/             # Centralized SUMO & project configuration
├── environment/        # SUMO network and route generation, TraCI wrapper
├── traffic_agents/     # PPO Agent implementation
├── models/             # PyTorch Actor-Critic models & saved checkpoints
├── training/           # Training loops and logging
├── evaluation/         # Baseline comparisons and metrics
├── dashboard/          # Streamlit visualization app
├── logs/               # TensorBoard logs & live metrics
├── plots/              # Result plots
├── demo/               # SUMO GUI demo scripts
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Python 3.10+**
- **SUMO Simulator:** [Download & Install SUMO](https://sumo.dlr.de/docs/Downloads.php).
- **Environment Variable:** Ensure `SUMO_HOME` is set to your SUMO installation directory.

### 2. Clone and Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate SUMO Network
```bash
python environment/generate_grid.py
```

---

## 🏋️ Training the Agents

To start the training process:
```bash
python training/train.py
```
This will:
1. Initialize the 6×6 grid environment.
2. Run PPO training for 100 episodes.
3. Save model checkpoints in `models/`.
4. Log metrics to TensorBoard.

**View Training Progress:**
```bash
tensorboard --logdir logs/
```

---

## 📊 Evaluation & Baselines

Compare the RL policy against Fixed-Time and Random controllers:
```bash
python evaluation/evaluate.py
```
This generates comparison plots in `plots/comparison_results.png`.

---

## 🖥️ Streamlit Dashboard

Launch the interactive dashboard to visualize performance:
```bash
streamlit run dashboard/streamlit_app.py
```

---

## 🎬 Real-Time Demo

Run the SUMO GUI to see the trained agent in action:
```bash
python demo/demo.py
```

---

## 🧠 Proximal Policy Optimization (PPO) Explanation

PPO is a state-of-the-art Policy Gradient method that strikes a balance between ease of implementation, sample efficiency, and ease of tuning.

- **Objective:** It prevents large, destabilizing policy updates by clipping the objective function.
- **Actor-Critic:** 
  - The **Actor** predicts the best action (traffic light phase change).
  - The **Critic** predicts the expected reward (value function), helping the actor learn more effectively.
- **Monte Carlo Returns:** Used to compute discounted rewards for advantage estimation.

---

## 📝 License
MIT License. See [LICENSE](LICENSE) for details.
