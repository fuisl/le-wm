"""Multi-signal real-SUMO environment (cologne8 and friends).

The topology-free counterpart to sumo_env.py (which is single-TL only). Exposes
the same *shape contract* the multi-agent JEPA pipeline already expects from
CTMGridEnv:

    state()            -> flat (N*F,) float32, F fixed across nodes
    encode_action(ph)  -> flat (N*A,) float32
    neighbor_table()   -> (idx (N, deg) int64, mask (N, deg) bool)
    node_feature_dim() -> F
    node_action_dim()  -> A
    step(phases)       -> next state()   (phases: (N,) int)
    save_state / load_state             (counterfactual branching)

Design choices (documented because they are the M1 "pad/pool to fixed F" step):

- Per-node feature F = 2*P_max + 1:
    * phase_pressure[p]  : summed halting-vehicle count on the lanes that green
                           phase p serves, p in [0, P_max). Zero-padded for TLs
                           with fewer than P_max phases. This is the
                           MaxPressure-shaped quantity the planner cares about
                           (sumo_env.controller_max_pressure uses the same sum).
    * phase_onehot[p]    : current green phase, one-hot, padded to P_max.
    * elapsed_norm       : time in current phase / 60 s.
  Lane count never enters F -> the same encoder runs on any intersection.

- Per-node action A = P_max: requested green phase, one-hot, padded.

- Neighbor table: TL j is a neighbour of TL i iff a network edge directly
  connects their junctions (either direction). Undirected adjacency, so pooling
  stays permutation-invariant; direction is left for the model to infer from
  the (upstream vs downstream) feature dynamics. deg = max adjacency over TLs.

- Phase actuation: any TL that changes phase spends the whole 5 s control step
  in transition (3 s yellow + 2 s all-red); TLs that hold keep their green for
  the full 5 s. Uniform and physically consistent; slightly pessimistic on the
  switch (new green starts next step). RESCO's create_yellows logic is reused
  verbatim from sumo_env.py.
"""

import os

import numpy as np
import sumolib

try:
    import libsumo as traci
    _USING_LIBSUMO = True
except ImportError:  # pragma: no cover
    import traci
    _USING_LIBSUMO = False

from traffic.sumo_env import create_yellows

STEP_LENGTH = 5
YELLOW_LENGTH = 3
CLEARANCE_LENGTH = 2


def _discover_green_phases(tl_id):
    """Green (non-yellow, not all-red) phases for a TL, in program order."""
    green = {}
    idx = 0
    logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
    for p in logic.getPhases():
        s = p.state
        if "y" not in s and (s.count("r") + s.count("s")) != len(s):
            green[idx] = s
            idx += 1
    return green


class SumoMultiEnv:
    EDGE_FEAT_DIM = 5          # per directed link: [occupancy, halting, speed/max, tt/fftt, veh count]
    EDGE_PAIR_DIM = 10        # per neighbour slot: [link j->i (5), link i->j (5)]

    def __init__(self, sumocfg_path, begin=25200, seed=0, warmup=10,
                 step_length=STEP_LENGTH, metrics=False, tripinfo_dir="/tmp/claude-1000",
                 hop_cap=999):
        self.hop_cap = hop_cap
        self.sumocfg_path = os.path.abspath(sumocfg_path)
        self.scenario_dir = os.path.dirname(self.sumocfg_path)
        self.begin = begin
        self.seed = int(seed)
        self.warmup = warmup
        self.step_length = step_length
        self.rng = np.random.default_rng(seed)

        self.metrics_on = metrics
        self._tripinfo_path = os.path.join(tripinfo_dir, f"_tripinfo_{os.getpid()}_{seed}.xml")
        self._q_accum = 0.0
        self._q_n = 0

        self._started = False
        self.tl_ids = []
        self.green_phases = {}      # tl_id -> {idx: state_str}
        self.yellow = {}           # tl_id -> create_yellows(...) dict
        self.controlled_lanes = {}  # tl_id -> [lane_id] aligned with state string
        self.current_phase = {}     # tl_id -> int
        self.elapsed = {}          # tl_id -> seconds in current phase
        self.P_max = 0
        self._nbr_idx = None
        self._nbr_mask = None
        self._parse_topology()

    # -- topology (static, from the .net.xml, no SUMO process needed) --------

    def _parse_topology(self):
        net_file = None
        # read net-file name out of the sumocfg
        for line in open(self.sumocfg_path):
            if "net-file" in line:
                net_file = line.split('value="')[1].split('"')[0]
                break
        net = sumolib.net.readNet(os.path.join(self.scenario_dir, net_file))
        tls_nodes = {}
        for tls in net.getTrafficLights():
            tid = tls.getID()
            try:
                node = net.getNode(tid)
            except KeyError:
                node = None
            tls_nodes[tid] = node
        self.tl_ids = sorted(tls_nodes)
        self._tl_index = {t: i for i, t in enumerate(self.tl_ids)}

        # adjacency: TL j is control-adjacent to TL i iff a directed chain of
        # edges runs from i's junction to j's junction WITHOUT passing through
        # any other TL junction. (Direct TL->TL edges are rare in these OSM-
        # derived nets - junctions are split by geometry nodes - so a plain
        # "edge between two TL nodes" test leaves almost every signal isolated.)
        node_to_tl = {n.getID(): t for t, n in tls_nodes.items() if n is not None}
        tl_node_ids = set(node_to_tl)
        # per-edge free-flow travel time and max speed, for normalising the edge state
        self._edge_maxspeed = {e.getID(): max(e.getSpeed(), 1.0) for e in net.getEdges()}
        self._edge_fftt = {e.getID(): max(e.getLength() / max(e.getSpeed(), 1.0), 1e-3)
                           for e in net.getEdges()}
        # adjacency + the FIRST edge leaving i's junction on the chain toward j
        # (the "link i->j" the coupling rides on). hop_cap keeps compact nets from
        # collapsing to near-complete graphs (ingolstadt21 mean degree 13.5 -> capped).
        adj = {t: set() for t in self.tl_ids}
        link_edges = {}     # (tl_i, tl_j) -> [sumo edge id(s), the first hop from i toward j]
        for tid, node in tls_nodes.items():
            if node is None:
                continue
            seen = {node.getID()}
            frontier = [(node, None)]   # (node, first-edge-id that started this chain)
            hops = 0
            while frontier and hops < self.hop_cap:
                nxt = []
                for nd, first_e in frontier:
                    for e in nd.getOutgoing():
                        tn = e.getToNode()
                        tnid = tn.getID()
                        fe = first_e if first_e is not None else e.getID()
                        if tnid in seen:
                            continue
                        seen.add(tnid)
                        if tnid in tl_node_ids and node_to_tl[tnid] != tid:
                            j = node_to_tl[tnid]
                            adj[tid].add(j)
                            adj[j].add(tid)
                            link_edges.setdefault((tid, j), []).append(fe)
                        else:
                            nxt.append((tn, fe))
                frontier = nxt
                hops += 1
        self._link_edges = link_edges
        deg = max((len(v) for v in adj.values()), default=0)
        deg = max(deg, 1)
        N = len(self.tl_ids)
        idx = -np.ones((N, deg), dtype=np.int64)
        mask = np.zeros((N, deg), dtype=bool)
        for t, nbrs in adj.items():
            i = self._tl_index[t]
            for k, nb in enumerate(sorted(nbrs)):
                idx[i, k] = self._tl_index[nb]
                mask[i, k] = True
        self._nbr_idx, self._nbr_mask = idx, mask
        self._adj = adj

    # -- lifecycle ---------------------------------------------------------

    def _start(self):
        if self._started:
            traci.close()
        args = [
            "sumo",
            "-c", self.sumocfg_path,
            "--begin", str(self.begin),
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--time-to-teleport", "-1",
            "--seed", str(self.seed),
        ]
        if self.metrics_on:
            args += [
                "--tripinfo-output", self._tripinfo_path,
                "--tripinfo-output.write-unfinished", "true",  # count vehicles still en route at end
                "--duration-log.statistics", "true",
            ]
        traci.start(args)
        self._started = True

    def reset(self):
        self._start()
        self._q_accum, self._q_n = 0.0, 0
        self.green_phases, self.yellow, self.controlled_lanes = {}, {}, {}
        self.current_phase, self.elapsed = {}, {}
        for tid in self.tl_ids:
            gp = _discover_green_phases(tid)
            self.green_phases[tid] = gp
            self.yellow[tid] = create_yellows(gp)
            self.controlled_lanes[tid] = list(traci.trafficlight.getControlledLanes(tid))
            self.current_phase[tid] = 0
            self.elapsed[tid] = 0
            traci.trafficlight.setRedYellowGreenState(tid, gp[0])
        self.P_max = max(len(gp) for gp in self.green_phases.values())

        for _ in range(self.warmup):
            self.step(np.array([self.current_phase[t] for t in self.tl_ids]))
        return self.state()

    def close(self):
        if self._started:
            traci.close()
            self._started = False

    # -- RESCO-style episode metrics -------------------------------------

    def metrics_tick(self):
        """Call once per control decision: accumulate instantaneous network
        queue (total halting vehicles on all controlled lanes)."""
        q = 0
        for tid in self.tl_ids:
            for lane in set(self.controlled_lanes[tid]):
                q += traci.lane.getLastStepHaltingNumber(lane)
        self._q_accum += q
        self._q_n += 1

    def episode_metrics(self):
        """Mean over all trips (finished + unfinished at episode end), matching
        RESCO's evaluation table:
          duration - trip travel time (s)          <- tripinfo `duration`
          delay    - time lost vs free-flow (s)    <- tripinfo `timeLoss`
          wait     - accumulated waiting time (s)  <- tripinfo `waitingTime`
          queue    - mean network halting vehicles per control step
          throughput - number of trips seen
        """
        import xml.etree.ElementTree as ET
        d = w = tl = 0.0
        n = 0
        try:
            for tr in ET.parse(self._tripinfo_path).getroot().findall("tripinfo"):
                d += float(tr.get("duration", 0.0))
                w += float(tr.get("waitingTime", 0.0))
                tl += float(tr.get("timeLoss", 0.0))
                n += 1
        except (FileNotFoundError, ET.ParseError):
            pass
        q = self._q_accum / self._q_n if self._q_n else float("nan")
        if n == 0:
            return {"duration": float("nan"), "delay": float("nan"),
                    "wait": float("nan"), "queue": q, "throughput": 0}
        return {"duration": d / n, "delay": tl / n, "wait": w / n,
                "queue": q, "throughput": n}

    # -- shapes ----------------------------------------------------------

    def node_feature_dim(self):
        return 2 * self.P_max + 1

    def node_action_dim(self):
        return self.P_max

    def n_nodes(self):
        return len(self.tl_ids)

    def state_dim(self):
        return self.n_nodes() * self.node_feature_dim()

    def action_dim(self):
        return self.n_nodes() * self.node_action_dim()

    def neighbor_table(self):
        return self._nbr_idx.copy(), self._nbr_mask.copy()

    def edge_pair_dim(self):
        return self.EDGE_PAIR_DIM

    # -- observation -----------------------------------------------------

    def _link_state(self, tl_a, tl_b):
        """State of the road link from TL a's junction toward TL b's junction:
        [occupancy, halting count, speed/maxspeed, traveltime/freeflow, veh count].
        Zeros if no such link. halting + veh count are left as raw counts (a
        platoon-in-transit proxy); the other three are 0..~1-ish ratios."""
        eids = self._link_edges.get((tl_a, tl_b), [])
        if not eids:
            return np.zeros(self.EDGE_FEAT_DIM, dtype=np.float32)
        occ = spd = ttr = 0.0
        hn = vn = 0.0
        for eid in eids:
            occ += traci.edge.getLastStepOccupancy(eid)
            spd += traci.edge.getLastStepMeanSpeed(eid) / self._edge_maxspeed.get(eid, 13.9)
            tt = traci.edge.getTraveltime(eid)
            ttr += tt / self._edge_fftt.get(eid, tt if tt > 0 else 1.0)
            hn += traci.edge.getLastStepHaltingNumber(eid)
            vn += traci.edge.getLastStepVehicleNumber(eid)
        n = len(eids)
        return np.array([occ / n, hn, spd / n, ttr / n, vn], dtype=np.float32)

    def edge_features(self):
        """(N, deg, EDGE_PAIR_DIM) aligned with neighbor_table(): for node i, slot
        k (neighbour j), [link j->i (what's arriving from j), link i->j (spillback
        room toward j)]. Zeros for masked slots."""
        N, deg = self._nbr_idx.shape
        out = np.zeros((N, deg, self.EDGE_PAIR_DIM), dtype=np.float32)
        for tid in self.tl_ids:
            i = self._tl_index[tid]
            for k in range(deg):
                j_idx = self._nbr_idx[i, k]
                if j_idx < 0:
                    continue
                jt = self.tl_ids[j_idx]
                out[i, k, : self.EDGE_FEAT_DIM] = self._link_state(jt, tid)   # j -> i (incoming)
                out[i, k, self.EDGE_FEAT_DIM:] = self._link_state(tid, jt)    # i -> j (outgoing)
        return out

    def _phase_pressure(self, tid):
        """Summed halting count on lanes served (green) by each green phase."""
        lanes = self.controlled_lanes[tid]
        halt = {}
        out = np.zeros(self.P_max, dtype=np.float32)
        for p, s in self.green_phases[tid].items():
            tot = 0.0
            for i, lane in enumerate(lanes):
                if i < len(s) and s[i] in ("G", "g"):
                    if lane not in halt:
                        halt[lane] = traci.lane.getLastStepHaltingNumber(lane)
                    tot += halt[lane]
            out[p] = tot
        return out

    def node_features(self):
        N, F = self.n_nodes(), self.node_feature_dim()
        feat = np.zeros((N, F), dtype=np.float32)
        for tid in self.tl_ids:
            i = self._tl_index[tid]
            feat[i, : self.P_max] = self._phase_pressure(tid)
            feat[i, self.P_max + self.current_phase[tid]] = 1.0
            feat[i, 2 * self.P_max] = self.elapsed[tid] / 60.0
        return feat

    def state(self):
        return self.node_features().reshape(-1)

    def pressure_slice(self):
        """Index range of the phase-pressure block inside each node's feature
        vector - the probe target / MaxPressure cost quantity."""
        return 0, self.P_max

    def encode_action(self, phases):
        phases = np.asarray(phases)
        oh = np.zeros((self.n_nodes(), self.P_max), dtype=np.float32)
        oh[np.arange(self.n_nodes()), phases] = 1.0
        return oh.reshape(-1)

    # -- save / restore ------------------------------------------------

    def save_state(self, path):
        traci.simulation.saveState(path)
        return {
            "path": path,
            "current_phase": dict(self.current_phase),
            "elapsed": dict(self.elapsed),
        }

    def load_state(self, snap):
        traci.simulation.loadState(snap["path"])
        self.current_phase = dict(snap["current_phase"])
        self.elapsed = dict(snap["elapsed"])
        for tid in self.tl_ids:
            traci.trafficlight.setRedYellowGreenState(
                tid, self.green_phases[tid][self.current_phase[tid]]
            )

    # -- stepping ------------------------------------------------------

    def step(self, phases):
        phases = np.asarray(phases).astype(int)
        switch = []
        for tid in self.tl_ids:
            i = self._tl_index[tid]
            want = int(np.clip(phases[i], 0, len(self.green_phases[tid]) - 1))
            cur = self.current_phase[tid]
            if want != cur:
                switch.append((tid, cur, want))

        if switch:
            for tid, cur, want in switch:
                key = f"{cur}_{want}"
                if key in self.yellow[tid]:
                    traci.trafficlight.setRedYellowGreenState(tid, self.yellow[tid][key])
            for _ in range(YELLOW_LENGTH):
                traci.simulationStep()
            for tid, cur, want in switch:
                traci.trafficlight.setRedYellowGreenState(tid, self.yellow[tid]["all_red"])
            for _ in range(CLEARANCE_LENGTH):
                traci.simulationStep()
            for tid, cur, want in switch:
                traci.trafficlight.setRedYellowGreenState(tid, self.green_phases[tid][want])
                self.current_phase[tid] = want
                self.elapsed[tid] = 0
            for tid in self.tl_ids:
                if tid not in [s[0] for s in switch]:
                    self.elapsed[tid] += self.step_length
        else:
            for _ in range(self.step_length):
                traci.simulationStep()
            for tid in self.tl_ids:
                self.elapsed[tid] += self.step_length

        return self.state()


# -- controllers, per-TL, returning a length-N phase vector -----------------

def controller_fixed_time(env, rng):
    cycles = {t: int(rng.integers(4, 10)) for t in env.tl_ids}
    offsets = {t: int(rng.integers(0, cycles[t])) for t in env.tl_ids}

    def policy(step):
        return np.array([
            ((step + offsets[t]) // cycles[t]) % len(env.green_phases[t])
            for t in env.tl_ids
        ])
    return policy


def controller_random(env, rng, switch_prob=0.25):
    state = {t: 0 for t in env.tl_ids}

    def policy(step):
        for t in env.tl_ids:
            if rng.random() < switch_prob:
                state[t] = int(rng.integers(0, len(env.green_phases[t])))
        return np.array([state[t] for t in env.tl_ids])
    return policy


def controller_max_pressure(env):
    def policy(step):
        out = []
        for tid in env.tl_ids:
            pr = env._phase_pressure(tid)[: len(env.green_phases[tid])]
            out.append(int(np.argmax(pr)) if pr.size else 0)
        return np.array(out)
    return policy
