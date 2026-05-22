"""
Service Queuing System Simulator
=================================
Run the simulation for a fixed duration, then display static graphs.

Usage examples:
  python queue_sim.py
  python queue_sim.py -n 4 --duration 30
  python queue_sim.py -n 2 --arrival-rate 2.0 --service-mean 3.0 --service-std 0.5
  python queue_sim.py --help

Parameters:
  -n / --servers        Number of servers (default: 3)
  --duration            How many seconds to simulate (default: 20)
  --arrival-rate        Average jobs arriving per second, e.g. 1.5 (Poisson process)
  --service-mean        Average seconds a server spends on one job, e.g. 2.0 (Normal dist)
  --service-std         Std-dev of service time (default: 0.5); set 0 for fixed service time
"""

import threading
import time
import random
import argparse
import queue

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np


# ─────────────────────────────────────────────────────────────
# Shared simulation state
# ─────────────────────────────────────────────────────────────
class SimState:
    def __init__(self, n_servers: int):
        self.n_servers       = n_servers
        self.job_queue       = queue.Queue(maxsize=200)
        self.lock            = threading.Lock()

        self.ts_time         = []   # snapshot timestamps (elapsed seconds)
        self.ts_queue_len    = []   # queue depth at each snapshot

        self.server_busy_sec = [0.0] * n_servers
        self.server_jobs     = [0]   * n_servers

        # Gantt: (server_id, start_elapsed, end_elapsed)
        self.gantt           = []
        self.gantt_lock      = threading.Lock()

        self.total_arrived   = 0
        self.total_served    = 0
        self.total_dropped   = 0

        self.start_time      = None
        self.running         = False


# ─────────────────────────────────────────────────────────────
# Worker threads
# ─────────────────────────────────────────────────────────────
def server_worker(server_id, state, service_mean, service_std):
    while state.running:
        try:
            job_id = state.job_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if service_std > 0:
            svc = max(0.05, random.gauss(service_mean, service_std))
        else:
            svc = service_mean

        t0 = time.time()
        time.sleep(svc)
        t1 = time.time()

        with state.lock:
            state.server_jobs[server_id]     += 1
            state.server_busy_sec[server_id] += (t1 - t0)
            state.total_served += 1

        rel_s = t0 - state.start_time
        rel_e = t1 - state.start_time
        with state.gantt_lock:
            state.gantt.append((server_id, rel_s, rel_e))

        state.job_queue.task_done()


def arrival_generator(state, arrival_rate):
    """Poisson arrivals: gaps between jobs ~ Exponential(1/arrival_rate)."""
    job_id = 0
    while state.running:
        gap = random.expovariate(arrival_rate)
        time.sleep(gap)
        if not state.running:
            break
        job_id += 1
        with state.lock:
            state.total_arrived += 1
        try:
            state.job_queue.put_nowait(job_id)
        except queue.Full:
            with state.lock:
                state.total_dropped += 1


def stats_collector(state, interval=0.25):
    """Snapshot queue depth every `interval` seconds."""
    while state.running:
        elapsed = time.time() - state.start_time
        qlen    = state.job_queue.qsize()
        with state.lock:
            state.ts_time.append(elapsed)
            state.ts_queue_len.append(qlen)
        time.sleep(interval)


# ─────────────────────────────────────────────────────────────
# Run simulation
# ─────────────────────────────────────────────────────────────
def run(n_servers, duration, arrival_rate, service_mean, service_std):
    state            = SimState(n_servers=n_servers)
    state.running    = True
    state.start_time = time.time()

    threads = []

    t = threading.Thread(target=arrival_generator,
                         args=(state, arrival_rate), daemon=True)
    t.start(); threads.append(t)

    t = threading.Thread(target=stats_collector, args=(state,), daemon=True)
    t.start(); threads.append(t)

    for i in range(n_servers):
        t = threading.Thread(target=server_worker,
                             args=(i, state, service_mean, service_std), daemon=True)
        t.start(); threads.append(t)

    time.sleep(duration)
    state.running = False

    # give in-flight jobs up to 2s to finish
    deadline = time.time() + 2.0
    while state.job_queue.qsize() > 0 and time.time() < deadline:
        time.sleep(0.05)

    return state


# ─────────────────────────────────────────────────────────────
# Static result graphs
# ─────────────────────────────────────────────────────────────
PALETTE = [
    "#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF",
    "#C77DFF", "#FF9F1C", "#2EC4B6", "#E71D36",
    "#F72585", "#4CC9F0", "#80B918", "#F4A261",
]


def plot_results(state, params):
    n       = state.n_servers
    col     = [PALETTE[i % len(PALETTE)] for i in range(n)]
    elapsed = state.ts_time[-1] if state.ts_time else params["duration"]

    fig = plt.figure(figsize=(14, 9), facecolor="#0f0f1a")
    fig.canvas.manager.set_window_title("Queue Simulation — Results")

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.6, wspace=0.35,
                           left=0.07, right=0.97,
                           top=0.88, bottom=0.08)

    ax_q    = fig.add_subplot(gs[0, :])
    ax_util = fig.add_subplot(gs[1, 0])
    ax_jobs = fig.add_subplot(gs[1, 1])
    ax_g    = fig.add_subplot(gs[2, :])

    for ax in (ax_q, ax_util, ax_jobs, ax_g):
        ax.set_facecolor("#14142b")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")
        ax.tick_params(colors="#aaaacc", labelsize=8)
        ax.xaxis.label.set_color("#aaaacc")
        ax.yaxis.label.set_color("#aaaacc")
        ax.title.set_color("#ddddff")

    drop_pct   = (state.total_dropped / state.total_arrived * 100) if state.total_arrived else 0
    throughput = state.total_served / elapsed if elapsed > 0 else 0

    fig.text(0.5, 0.95,
             f"SERVICE QUEUE  |  {n} server{'s' if n > 1 else ''}  |  "
             f"arrival={params['arrival_rate']}/s  "
             f"service={params['service_mean']}s +/- {params['service_std']}s  |  "
             f"duration={params['duration']}s",
             ha="center", color="#ffffff", fontsize=11,
             fontweight="bold", fontfamily="monospace")

    fig.text(0.5, 0.915,
             f"arrived={state.total_arrived}   served={state.total_served}   "
             f"dropped={state.total_dropped} ({drop_pct:.1f}%)   "
             f"throughput={throughput:.2f} jobs/s",
             ha="center", color="#aaaaff", fontsize=9, fontfamily="monospace")

    # ── 1. Queue length over time ──────────────────────────
    ax_q.set_title("Queue Length over Time", fontsize=10)
    ax_q.set_xlabel("Elapsed (s)")
    ax_q.set_ylabel("Jobs waiting")
    if state.ts_time:
        ax_q.fill_between(state.ts_time, state.ts_queue_len,
                           alpha=0.3, color="#4D96FF")
        ax_q.plot(state.ts_time, state.ts_queue_len,
                  color="#4D96FF", linewidth=1.5)
        ax_q.set_xlim(0, elapsed)
        peak = max(state.ts_queue_len) if state.ts_queue_len else 1
        ax_q.set_ylim(0, peak + 2)

    # ── 2. Server utilisation bars ─────────────────────────
    ax_util.set_title("Server Utilisation (%)", fontsize=10)
    ax_util.set_ylabel("Utilisation %")
    ax_util.set_ylim(0, 115)
    utils = [min(100.0, (b / elapsed) * 100) for b in state.server_busy_sec]
    xpos  = np.arange(n)
    bars  = ax_util.bar(xpos, utils, color=col, width=0.6,
                         alpha=0.85, edgecolor="#ffffff22")
    for bar, u in zip(bars, utils):
        ax_util.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 2, f"{u:.1f}%",
                     ha="center", va="bottom", color="#ffffff",
                     fontsize=8, fontweight="bold")
    ax_util.axhline(100, color="#FF6B6B", linewidth=0.8,
                    linestyle="--", alpha=0.5)
    ax_util.set_xticks(xpos)
    ax_util.set_xticklabels([f"S{i}" for i in range(n)], fontsize=8)

    # ── 3. Jobs completed per server ───────────────────────
    ax_jobs.set_title("Jobs Completed per Server", fontsize=10)
    ax_jobs.set_ylabel("Jobs")
    bars2 = ax_jobs.bar(xpos, state.server_jobs, color=col, width=0.6,
                         alpha=0.85, edgecolor="#ffffff22")
    for bar, j in zip(bars2, state.server_jobs):
        ax_jobs.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.3, str(j),
                     ha="center", va="bottom", color="#ffffff",
                     fontsize=8, fontweight="bold")
    ax_jobs.set_xticks(xpos)
    ax_jobs.set_xticklabels([f"S{i}" for i in range(n)], fontsize=8)
    peak_jobs = max(state.server_jobs, default=1)
    ax_jobs.set_ylim(0, peak_jobs * 1.25 + 1)

    # ── 4. Gantt chart ─────────────────────────────────────
    ax_g.set_title("Server Activity — Gantt Chart", fontsize=10)
    ax_g.set_xlabel("Elapsed (s)")
    ax_g.set_ylabel("Server")
    with state.gantt_lock:
        events = list(state.gantt)
    for sid, s, e in events:
        ax_g.barh(sid, e - s, left=s, height=0.55,
                  color=col[sid % len(col)], alpha=0.8,
                  edgecolor="#ffffff11")
    ax_g.set_xlim(0, elapsed)
    ax_g.set_ylim(-0.6, n - 0.4)
    ax_g.set_yticks(range(n))
    ax_g.set_yticklabels([f"S{i}" for i in range(n)], fontsize=8)

    patches = [mpatches.Patch(color=col[i], label=f"Server {i}")
               for i in range(n)]
    fig.legend(handles=patches, loc="lower center",
               ncol=min(n, 10), fontsize=8, framealpha=0.15,
               labelcolor="#ccccff", bbox_to_anchor=(0.5, 0.005))

    plt.show()


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Service Queue Simulator — runs for a fixed time, then shows graphs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
HOW RANDOMIZATION WORKS
------------------------
  --arrival-rate (lambda):
      Jobs arrive randomly. On average, `lambda` jobs arrive per second.
      The gaps between arrivals are random (Poisson process / exponential gaps).
      Example: --arrival-rate 2.0  =>  roughly 2 jobs per second on average.

  --service-mean (mu):
      Each job takes a random amount of time. The average is `mu` seconds.
      Example: --service-mean 3.0  =>  jobs take about 3 seconds on average.

  --service-std (sigma):
      How spread out the service times are. Larger = more variability.
      Set to 0 for every job to take exactly `mu` seconds (no randomness).
      Example: --service-std 0.5  =>  most jobs take between 1.5s and 4.5s (with mu=3).
      Example: --service-std 0    =>  every job takes exactly mu seconds.

TRAFFIC LOAD (rho = arrival_rate / (servers x (1/service_mean))):
  rho < 1  -> system is stable, queue stays bounded
  rho >= 1 -> system is overloaded, queue grows without limit

EXAMPLES
--------
  python queue_sim.py
  python queue_sim.py -n 5 --duration 30
  python queue_sim.py -n 2 --arrival-rate 3.0 --service-mean 2.0 --service-std 0.8
  python queue_sim.py -n 1 --arrival-rate 0.5 --service-mean 1.0 --service-std 0
        """,
    )
    parser.add_argument("-n", "--servers",
                        type=int, default=3, metavar="N",
                        help="Number of servers, must be > 0 (default: 3)")
    parser.add_argument("--duration",
                        type=float, default=20, metavar="SEC",
                        help="How many seconds to run the simulation (default: 20)")
    parser.add_argument("--arrival-rate",
                        type=float, default=1.5, metavar="LAMBDA",
                        help="Average job arrivals per second (default: 1.5)")
    parser.add_argument("--service-mean",
                        type=float, default=2.0, metavar="MU",
                        help="Average service time per job in seconds (default: 2.0)")
    parser.add_argument("--service-std",
                        type=float, default=0.5, metavar="SIGMA",
                        help="Std-dev of service time; 0 = fixed/no randomness (default: 0.5)")
    args = parser.parse_args()

    if args.servers < 1:
        parser.error("-n / --servers must be > 0")
    if args.duration <= 0:
        parser.error("--duration must be > 0")
    if args.arrival_rate <= 0:
        parser.error("--arrival-rate must be > 0")
    if args.service_mean <= 0:
        parser.error("--service-mean must be > 0")
    if args.service_std < 0:
        parser.error("--service-std must be >= 0")

    rho = args.arrival_rate / (args.servers / args.service_mean)
    print(f"\n{'='*58}")
    print(f"  SERVICE QUEUE SIMULATOR")
    print(f"{'='*58}")
    print(f"  Servers            : {args.servers}")
    print(f"  Duration           : {args.duration}s")
    print(f"  Arrival rate       : {args.arrival_rate} jobs/s  (Poisson/random)")
    print(f"  Service time mean  : {args.service_mean}s  (Normal/random)")
    print(f"  Service time std   : {args.service_std}s  {'(fixed — no randomness)' if args.service_std == 0 else ''}")
    print(f"  Traffic load (rho) : {rho:.2f}  {'<-- OVERLOADED' if rho >= 1 else '(stable)'}")
    print(f"{'='*58}")
    print(f"  Simulating... ", end="", flush=True)

    state = run(
        n_servers    = args.servers,
        duration     = args.duration,
        arrival_rate = args.arrival_rate,
        service_mean = args.service_mean,
        service_std  = args.service_std,
    )

    elapsed    = state.ts_time[-1] if state.ts_time else args.duration
    drop_pct   = (state.total_dropped / state.total_arrived * 100) if state.total_arrived else 0
    throughput = state.total_served / elapsed if elapsed > 0 else 0

    print("done.")
    print(f"\n  Arrived    : {state.total_arrived}")
    print(f"  Served     : {state.total_served}")
    print(f"  Dropped    : {state.total_dropped} ({drop_pct:.1f}%)")
    print(f"  Throughput : {throughput:.2f} jobs/s")
    print(f"\n  Opening graphs...\n")

    plot_results(state, {
        "arrival_rate": args.arrival_rate,
        "service_mean": args.service_mean,
        "service_std":  args.service_std,
        "duration":     args.duration,
    })


if __name__ == "__main__":
    main()