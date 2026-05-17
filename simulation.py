import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Reproducibility
random.seed(42)
np.random.seed(42)

# Load dataset
df = pd.read_csv("/home/hp/edgeproject/data.csv", sep=",", engine="python")
df = df.head(10000)
print(f"Loaded {len(df)} tasks")

# Define edge nodes
NUM_NODES = 10
WORKERS_PER_NODE = 2

def make_nodes():
    return {
        f"node-{i}": {"id": f"node-{i}", "slots": [0.0] * WORKERS_PER_NODE}
        for i in range(NUM_NODES)
    }

def get_nearest_node(device_id):
    device_num = int(device_id.split("-")[1]) % NUM_NODES
    return f"node-{device_num}"

def get_processing_time():
    return np.random.lognormal(mean=3.5, sigma=0.9)

def assign_to_node(node, arrival):
    slot_idx = int(np.argmin(node["slots"]))
    start_time = max(arrival, node["slots"][slot_idx])
    finish = start_time + get_processing_time()
    node["slots"][slot_idx] = finish
    return finish

def get_node_load(node):
    return min(node["slots"])

# ── SIMULATION CORE ──────────────────────────────────────────────────────────

def simulate(df, placement="nearest", redundancy="none"):
    nodes = make_nodes()
    results = []

    for _, task in df.iterrows():
        arrival = task["arrival_ms"]
        deadline = task["deadline_ms"]

        if placement == "nearest":
            chosen = get_nearest_node(task["device_id"])
        elif placement == "p2c":
            sample = random.sample(list(nodes.keys()), 2)
            chosen = min(sample, key=lambda n: get_node_load(nodes[n]))

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

        actual_finish = min(finish_times)
        latency = actual_finish - arrival
        results.append({
            "req_id": task["req_id"],
            "latency_ms": latency,
            "sla_violated": latency > deadline,
            "replicas": len(finish_times)
        })

    return pd.DataFrame(results)

# ── RUN ALL SCENARIOS ─────────────────────────────────────────────────────────

scenarios = [
    ("nearest", "none",   "nearest\n+none"),
    ("p2c",     "none",   "p2c\n+none"),
    ("p2c",     "fixed2", "p2c\n+fixed2"),
    ("p2c",     "delayed","p2c\n+delayed"),
]

all_results = {}
labels = []
p95_values = []
p99_values = []
violation_pcts = []
avg_replica_values = []

print(f"\n{'Scenario':<30} {'P95':>10} {'P99':>10} {'SLA Violations':>15} {'Avg Replicas':>13}")
print("-" * 80)

for placement, redundancy, label in scenarios:
    random.seed(42)
    np.random.seed(42)
    results = simulate(df, placement=placement, redundancy=redundancy)
    p95 = results["latency_ms"].quantile(0.95)
    p99 = results["latency_ms"].quantile(0.99)
    violations = results["sla_violated"].sum()
    violation_pct = results["sla_violated"].mean() * 100
    avg_replicas = results["replicas"].mean()

    all_results[label] = results
    labels.append(label)
    p95_values.append(p95)
    p99_values.append(p99)
    violation_pcts.append(violation_pct)
    avg_replica_values.append(avg_replicas)

    print(f"{label.replace(chr(10), ' '):<30} {p95:>10.2f} {p99:>10.2f} "
          f"{violations:>8} ({violation_pct:>4.1f}%) {avg_replicas:>13.2f}")

# ── CHART 1: P95 AND P99 LATENCY ─────────────────────────────────────────────

x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, p95_values, width, label="P95 Latency", color="yellow")
ax.bar(x + width/2, p99_values, width, label="P99 Latency", color="green")
ax.set_ylabel("Latency (ms)")
ax.set_title("P95 and P99 Latency by Strategy")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_yscale("log")
plt.tight_layout()
plt.savefig("/home/hp/edgeproject/chart_latency.png", dpi=150)
print("Saved chart_latency.png")

# ── CHART 2: SLA VIOLATION RATE ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(labels, violation_pcts,
              color=["steelblue", "seagreen", "tomato", "gold"])
ax.set_ylabel("SLA Violation Rate (%)")
ax.set_title("SLA Violation Rate by Strategy")
for bar, val in zip(bars, violation_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig("/home/hp/edgeproject/chart_sla.png", dpi=150)
print("Saved chart_sla.png")

# ── CHART 3: LATENCY DISTRIBUTION ────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))
colors = ["steelblue", "seagreen", "gold"]
color_idx = 0
for label, results in all_results.items():
    if "fixed2" not in label:
        ax.hist(results["latency_ms"].clip(upper=1000), bins=100,
                alpha=0.5, label=label, color=colors[color_idx])
        color_idx += 1
ax.set_xlabel("Latency (ms)")
ax.set_ylabel("Number of Tasks")
ax.set_title("Latency Distribution (clipped at 1000ms)")
ax.legend()
plt.tight_layout()
plt.savefig("/home/hp/edgeproject/chart_distribution.png", dpi=150)
print("Saved chart_distribution.png")

# ── CHART 4: AVERAGE REPLICAS (RESOURCE COST) ────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(labels, avg_replica_values,
              color=["steelblue", "seagreen", "tomato", "gold"])
ax.set_ylabel("Average Replicas per Task")
ax.set_title("Resource Cost by Strategy (Avg Replicas)")
ax.set_ylim(0, 2.5)
for bar, val in zip(bars, avg_replica_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val:.2f}", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig("/home/hp/edgeproject/chart_replicas.png", dpi=150)
print("Saved chart_replicas.png")

# ── CHART 5: P99 VS SLA VIOLATIONS COMBINED ──────────────────────────────────

filtered = [(l, p, v) for l, p, v in zip(labels, p99_values, violation_pcts)
            if "fixed2" not in l]
f_labels, f_p99, f_viol = zip(*filtered)
x2 = np.arange(len(f_labels))
width = 0.35

fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

ax1.bar(x2 - width/2, f_p99, width, color="steelblue", label="P99 Latency")
ax2.bar(x2 + width/2, f_viol, width, color="tomato", alpha=0.7, label="SLA Violation %")

ax1.set_ylabel("P99 Latency (ms)", color="steelblue")
ax2.set_ylabel("SLA Violation Rate (%)", color="tomato")
ax1.set_title("P99 Latency vs SLA Violation Rate (excluding fixed2)")
ax1.set_xticks(x2)
ax1.set_xticklabels(f_labels)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.tight_layout()
plt.savefig("/home/hp/edgeproject/chart_p99_vs_sla.png", dpi=150)
print("Saved chart_p99_vs_sla.png")
