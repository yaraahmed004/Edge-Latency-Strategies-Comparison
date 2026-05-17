import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Reproducibility
random.seed(42)
np.random.seed(42)

# Load dataset
df = pd.read_csv("/home/hp/edgeproject/data.csv", sep=",", engine="python")
print(f"Loaded {len(df)} tasks")

# Define edge nodes
NUM_NODES = 10
WORKERS_PER_NODE = 2  # each node can process 5 tasks concurrently

def make_nodes():
    return {
        f"node-{i}": {
            "id": f"node-{i}",
            "slots": [0.0] * WORKERS_PER_NODE  # each slot tracks when it's free
        }
        for i in range(NUM_NODES)
    }

def get_nearest_node(device_id):
    device_num = int(device_id.split("-")[1]) % NUM_NODES
    return f"node-{device_num}"

def get_processing_time():
    return np.random.lognormal(mean=3.5, sigma=0.9)

def assign_to_node(node, arrival):
    # pick the slot that's free earliest
    slot_idx = int(np.argmin(node["slots"]))
    start_time = max(arrival, node["slots"][slot_idx])
    proc_time = get_processing_time()
    finish = start_time + proc_time
    node["slots"][slot_idx] = finish
    return finish

def get_node_load(node):
    # load = earliest time any slot is free
    return min(node["slots"])

# -------- ML TRAINING -------------------------------
def generate_training_data(df):
    nodes = make_nodes()
    training_data = []

    for _, task in df.iterrows():
        arrival = task["arrival_ms"]
        deadline = task["deadline_ms"]

        # get chosen node via p2c
        sample = random.sample(list(nodes.keys()), 2)
        chosen = min(sample, key=lambda n: get_node_load(nodes[n]))

        # record cluster state BEFORE assigning
        avg_load = np.mean([get_node_load(nodes[n]) for n in nodes])
        chosen_load = get_node_load(nodes[chosen])

        # simulate single replica
        node = nodes[chosen]
        slot_idx = int(np.argmin(node["slots"]))
        start_time = max(arrival, node["slots"][slot_idx])
        proc_time = get_processing_time()
        finish_single = start_time + proc_time
        node["slots"][slot_idx] = finish_single

        latency_single = finish_single - arrival
        sla_violated_single = latency_single > deadline

        # label: would a second replica have helped?
        # try a second node
        others = [n for n in nodes if n != chosen]
        second = min(others, key=lambda n: get_node_load(nodes[n]))
        node2 = nodes[second]
        slot_idx2 = int(np.argmin(node2["slots"]))
        start_time2 = max(arrival, node2["slots"][slot_idx2])
        proc_time2 = get_processing_time()
        finish_second = start_time2 + proc_time2

        # label = 1 if second replica would have finished faster and saved the SLA
        best_finish = min(finish_single, finish_second)
        latency_best = best_finish - arrival
        label = 1 if latency_single > 200 else 0

        training_data.append({
            "avg_load": avg_load,
            "chosen_load": chosen_load,
            "arrival": arrival,
            "latency_single": latency_single,
            "sla_violated": int(sla_violated_single),
            "label": label
        })

    return pd.DataFrame(training_data)

# ── SIMULATION CORE ──────────────────────────────────────────────────────────

def simulate(df, placement="nearest", redundancy="none"):
    nodes = make_nodes()
    results = []

    for _, task in df.iterrows():
        arrival = task["arrival_ms"]
        deadline = task["deadline_ms"]

        # --- Placement decision ---
        if placement == "nearest":
            chosen = get_nearest_node(task["device_id"])

        elif placement == "p2c":
            sample = random.sample(list(nodes.keys()), 2)
            chosen = min(sample, key=lambda n: get_node_load(nodes[n]))

        # --- Redundancy decision & execution ---
        if redundancy == "none":
            finish_times = [assign_to_node(nodes[chosen], arrival)]

        elif redundancy == "fixed2":
            others = [n for n in nodes if n != chosen]
            second = random.choice(others)
            finish_times = [
                assign_to_node(nodes[chosen], arrival),
                assign_to_node(nodes[second], arrival)
            ]

        elif redundancy == "delayed":
            finish_times = [assign_to_node(nodes[chosen], arrival)]
            if finish_times[0] - arrival > 200:
                others = [n for n in nodes if n != chosen]
                hedge = min(others, key=lambda n: get_node_load(nodes[n]))
                finish_times.append(assign_to_node(nodes[hedge], arrival))

        # First finish wins
        actual_finish = min(finish_times)
        latency = actual_finish - arrival
        sla_violated = latency > deadline

        results.append({
            "req_id": task["req_id"],
            "latency_ms": latency,
            "sla_violated": sla_violated,
            "replicas": len(finish_times)
        })

    return pd.DataFrame(results)

# ── RUN ALL SCENARIOS ─────────────────────────────────────────────────────────

scenarios = [
    ("nearest", "none"),
    ("p2c",     "none"),
    ("p2c",     "fixed2"),
    ("p2c",     "delayed"),
    ("nearest", "fixed2"),
]

print(f"\n{'Scenario':<30} {'P95':>10} {'P99':>10} {'SLA Violations':>15} {'Avg Replicas':>13}")
print("-" * 80)

for placement, redundancy in scenarios:
    random.seed(42)
    np.random.seed(42)
    results = simulate(df, placement=placement, redundancy=redundancy)
    p95 = results["latency_ms"].quantile(0.95)
    p99 = results["latency_ms"].quantile(0.99)
    violations = results["sla_violated"].sum()
    violation_pct = results["sla_violated"].mean() * 100
    avg_replicas = results["replicas"].mean()
    label = f"{placement} + {redundancy}"
    print(f"{label:<30} {p95:>10.2f} {p99:>10.2f} {violations:>8} ({violation_pct:>4.1f}%) {avg_replicas:>13.2f}")

# ── COLLECT ALL RESULTS ───────────────────────────────────────────────────────

scenarios = [
    ("nearest", "none"),
    ("p2c",     "none"),
    ("p2c",     "fixed2"),
    ("p2c",     "delayed"),
]

labels = []
p95_values = []
p99_values = []
violation_pcts = []
all_results = {}

for placement, redundancy in scenarios:
    random.seed(42)
    np.random.seed(42)
    results = simulate(df, placement=placement, redundancy=redundancy)
    label = f"{placement}\n+{redundancy}"
    labels.append(label)
    p95_values.append(results["latency_ms"].quantile(0.95))
    p99_values.append(results["latency_ms"].quantile(0.99))
    violation_pcts.append(results["sla_violated"].mean() * 100)
    all_results[label] = results

# ── CHART 1: P95 AND P99 LATENCY ─────────────────────────────────────────────

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, p95_values, width, label="P95 Latency", color="steelblue")
bars2 = ax.bar(x + width/2, p99_values, width, label="P99 Latency", color="tomato")
ax.set_ylabel("Latency (ms)")
ax.set_title("P95 and P99 Latency by Strategy")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_yscale("log")  # log scale so fixed2 doesn't dwarf everything
plt.tight_layout()
plt.savefig("/home/hp/edgeproject/chart_latency.png", dpi=150)
print("Saved chart_latency.png")

# ── CHART 2: SLA VIOLATION RATE ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, violation_pcts, color=["steelblue", "seagreen", "tomato", "gold"])
ax.set_ylabel("SLA Violation Rate (%)")
ax.set_title("SLA Violation Rate by Strategy")
for bar, val in zip(bars, violation_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig("/home/hp/edgeproject/chart_sla.png", dpi=150)
print("Saved chart_sla.png")

# ── CHART 3: LATENCY DISTRIBUTION ────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))
for label, results in all_results.items():
    if "fixed2" not in label:  # skip fixed2 for distribution chart
        clipped = results["latency_ms"].clip(upper=1000)
        ax.hist(clipped, bins=100, alpha=0.5, label=label)
ax.set_xlabel("Latency (ms)")
ax.set_ylabel("Number of Tasks")
ax.set_title("Latency Distribution (clipped at 1000ms)")
ax.legend()
plt.tight_layout()
plt.savefig("/home/hp/edgeproject/chart_distribution.png", dpi=150)
print("Saved chart_distribution.png")

# Generate training data
print("\nGenerating training data...")
random.seed(42)
np.random.seed(42)
train_df = generate_training_data(df)
print(f"Training data shape: {train_df.shape}")
print(f"Label distribution: {train_df['label'].value_counts().to_dict()}")
