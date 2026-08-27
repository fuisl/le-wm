"""Minimal store-and-forward / CTM-style traffic network on a grid of intersections.

Each intersection has 4 approach queues (N, S, E, W), one per compass heading.
queue[i, d] holds vehicles at intersection i travelling in heading d, about to
be discharged toward neighbor(i, d) if the signal is green for that movement
(through-traffic only, no turns). Two signal phases per intersection: phase 0
discharges N/S, phase 1 discharges E/W. Outflow from one intersection becomes
inflow to its downstream neighbor's same-heading queue, capped by that
neighbor's spare capacity (spillback). Queues on the grid boundary receive
stochastic external arrivals; outflow that exits the grid leaves the network.

This is deliberately not a full CTM (single queue per approach, not a chain of
cells) - it's the sanity-ceiling generator: known ground-truth dynamics with
real multi-agent coupling (spillback) and a real action -> future-state causal
link, so a JEPA world model can be validated before touching a real simulator.
"""

import numpy as np

# direction index -> (row delta, col delta)
DIRS = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}  # N, S, E, W
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}


class CTMGridEnv:
    def __init__(self, rows=2, cols=2, capacity=20.0, sat_flow=2.0,
                 arrival_rate=0.35, seed=None):
        self.rows, self.cols = rows, cols
        self.n = rows * cols
        self.capacity = capacity
        self.sat_flow = sat_flow
        self.arrival_rate = arrival_rate
        self.rng = np.random.default_rng(seed)
        self.queues = None  # (n, 4) float
        self.phase = None   # (n,) int in {0, 1}
        self.t = 0

    def idx(self, r, c):
        return r * self.cols + c

    def neighbor(self, r, c, d):
        dr, dc = DIRS[d]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.rows and 0 <= nc < self.cols:
            return self.idx(nr, nc)
        return None

    def state_dim(self):
        return 4 * self.n

    def action_dim(self):
        return 2 * self.n

    def node_feature_dim(self):
        """Fixed per-node feature width F, independent of grid size (n).

        On this compass grid, F is fixed for free (4 approach queues per
        node) - no pad/pool-over-lanes machinery needed here. That machinery
        (M1: pad/pool per-lane queue/count/speed/occupancy to fixed F) is
        only earned when lane count varies across nodes, i.e. real SUMO
        intersections, not this synthetic grid.
        """
        return 4

    def neighbor_table(self):
        """Directed N/S/E/W neighbor ids for every node, fixed shape (n, 4).

        idx[i, d] is the node id reached from i in direction d (see DIRS),
        or -1 if i is on the boundary in that direction. mask[i, d] is True
        iff that neighbor exists. Use mask (not idx == -1) as the pooling
        weight so boundary nodes get a mean over their *actual* neighbors,
        not a zero-padded-then-divided-by-4 value - the latter leaks
        corner/edge/interior identity into the pooled feature, which is
        exactly the scenario-specific signal this is meant to avoid.
        """
        idx = -np.ones((self.n, 4), dtype=np.int64)
        mask = np.zeros((self.n, 4), dtype=bool)
        for r in range(self.rows):
            for c in range(self.cols):
                i = self.idx(r, c)
                for d in range(4):
                    j = self.neighbor(r, c, d)
                    if j is not None:
                        idx[i, d] = j
                        mask[i, d] = True
        return idx, mask

    def node_features(self):
        """Per-node observation, fixed shape (n, F). F == node_feature_dim()."""
        return self.queues.astype(np.float32)

    def node_action(self, phases):
        """Per-node action one-hot, fixed shape (n, 2)."""
        phases = np.asarray(phases)
        onehot = np.zeros((self.n, 2), dtype=np.float32)
        onehot[np.arange(self.n), phases] = 1.0
        return onehot

    def reset(self):
        self.queues = self.rng.uniform(0, self.capacity * 0.3, size=(self.n, 4))
        self.phase = self.rng.integers(0, 2, size=self.n)
        self.t = 0
        return self.state()

    def state(self):
        return self.queues.flatten().astype(np.float32)

    def encode_action(self, phases):
        """phases: (n,) int in {0,1} -> one-hot (2n,) float32."""
        onehot = np.zeros((self.n, 2), dtype=np.float32)
        onehot[np.arange(self.n), np.asarray(phases)] = 1.0
        return onehot.flatten()

    def clone_state(self):
        return (self.queues.copy(), self.phase.copy(), self.t)

    def set_state(self, snapshot):
        q, p, t = snapshot
        self.queues = q.copy()
        self.phase = p.copy()
        self.t = t

    def step(self, phases):
        """Advance one control step under the given per-intersection phase choice.

        phases: (n,) int array in {0, 1}.
        """
        phases = np.asarray(phases)
        n, C = self.n, self.cols
        outflow = np.zeros((n, 4))

        for i in range(n):
            green_dirs = (0, 1) if phases[i] == 0 else (2, 3)
            for d in green_dirs:
                outflow[i, d] = min(self.sat_flow, self.queues[i, d])

        inflow = np.zeros((n, 4))
        for i in range(n):
            r, c = divmod(i, C)
            for d in range(4):
                if outflow[i, d] <= 0:
                    continue
                j = self.neighbor(r, c, d)
                if j is None:
                    continue  # exits the network
                spare = max(0.0, self.capacity - self.queues[j, d] - inflow[j, d])
                actual = min(outflow[i, d], spare)
                outflow[i, d] = actual  # demand that can't be received stays queued
                inflow[j, d] += actual

        self.queues = self.queues - outflow + inflow

        for i in range(n):
            r, c = divmod(i, C)
            for d in range(4):
                if self.neighbor(r, c, OPPOSITE[d]) is None:
                    arrivals = self.rng.poisson(self.arrival_rate)
                    self.queues[i, d] += arrivals

        self.queues = np.clip(self.queues, 0, self.capacity)
        self.phase = phases
        self.t += 1
        return self.state()


def controller_fixed_time(env, cycle, offset):
    def policy(t):
        return (((t + offset) // cycle) % 2).astype(int)
    return policy


def controller_random(env, switch_prob, rng=None):
    rng = rng if rng is not None else env.rng
    phase = env.phase.copy()

    def policy(t):
        nonlocal phase
        switch = rng.random(env.n) < switch_prob
        phase = np.where(switch, 1 - phase, phase)
        return phase.copy()
    return policy


def controller_max_pressure(env):
    def policy(t):
        ns_pressure = env.queues[:, 0] + env.queues[:, 1]
        ew_pressure = env.queues[:, 2] + env.queues[:, 3]
        return (ew_pressure > ns_pressure).astype(int)
    return policy
