"""Real-SUMO traffic-signal environment, same interface shape as ctm_env.CTMGridEnv
(reset/step/state/encode_action/clone_state/set_state) so it drops into the same
dataset generation, training, and eval code unchanged.

Drives a RESCO (https://github.com/Pi-Star-Lab/RESCO) scenario directly via
libsumo/traci rather than RESCO's own gym wrapper, to avoid taking on its full
config/multi-signal stack for a first single-intersection pass. Signal-phase
timing (green-phase discovery, yellow transition construction, all-red
clearance) follows RESCO's own conventions (step_length=5s, yellow_length=3s,
clearance_length=2s) so the actuation model is physically realistic rather
than an instant phase jump.

Currently single-traffic-light scenarios only (e.g. cologne1). Multi-signal
scenarios (cologne3/8, grid4x4) need per-signal state/action concatenation,
not yet implemented.
"""

import itertools

import numpy as np

try:
    import libsumo as traci
    _USING_LIBSUMO = True
except ImportError:
    import traci
    _USING_LIBSUMO = False

STEP_LENGTH = 5      # seconds per control decision
YELLOW_LENGTH = 3    # seconds of yellow when switching green phases
CLEARANCE_LENGTH = 2  # seconds of all-red after a permissive ('g') phase


def create_yellows(phases):
    """Verbatim logic from resco_benchmark/traffic_signal.py: current-phase pairs
    -> the yellow signal string for the transition between them."""
    yellow_transitions = {}
    for current_phase in range(len(phases)):
        for next_phase in range(len(phases)):
            intermediate = []
            current_colors = phases[current_phase]
            next_colors = phases[next_phase]
            for i, color in enumerate(current_colors):
                next_color = next_colors[i]
                green = color in ("G", "g")
                red_next = next_color in ("r", "s")
                if green and red_next:
                    intermediate.append("y")
                else:
                    intermediate.append(color)
            if "y" in intermediate:
                yellow_transitions[f"{current_phase}_{next_phase}"] = "".join(intermediate)
        all_red = "r" * len(phases[current_phase])
        yellow_transitions["all_red"] = all_red
    return yellow_transitions


class SumoTLEnv:
    def __init__(self, sumocfg_path, tl_id=None, seed=None, warmup=0):
        self.sumocfg_path = sumocfg_path
        self.tl_id_arg = tl_id
        self.seed = seed if seed is not None else 0
        self.warmup = warmup

        self.tl_id = None
        self.lanes = []
        self.green_phases = {}
        self.yellow_transitions = {}
        self.current_phase = 0
        self.rng = np.random.default_rng(seed)
        self._started = False

    # -- lifecycle -----------------------------------------------------

    def _start_sumo(self):
        if self._started:
            traci.close()
        args = [
            "-c", self.sumocfg_path,
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--time-to-teleport", "-1",
            "--seed", str(self.seed),
        ]
        if _USING_LIBSUMO:
            traci.start(["sumo"] + args)
        else:
            traci.start(["sumo"] + args)
        self._started = True

    def reset(self):
        self._start_sumo()

        self.tl_id = self.tl_id_arg or traci.trafficlight.getIDList()[0]

        self.green_phases = {}
        idx = 0
        for p in traci.trafficlight.getAllProgramLogics(self.tl_id)[0].getPhases():
            state = p.state
            if "y" not in state and state.count("r") + state.count("s") != len(state):
                self.green_phases[idx] = state
                idx += 1
        self.yellow_transitions = create_yellows(self.green_phases)

        # NOT deduplicated: one entry per controlled *link*, same order as each
        # green_phases[i] state string, so state()[k] and phase_state[k] refer
        # to the same link (a lane with two movements appears twice, by design).
        self.lanes = list(traci.trafficlight.getControlledLanes(self.tl_id))

        self.current_phase = 0
        traci.trafficlight.setRedYellowGreenState(self.tl_id, self.green_phases[0])

        for _ in range(self.warmup):
            traci.simulationStep()

        return self.state()

    def close(self):
        if self._started:
            traci.close()
            self._started = False

    # -- shapes ----------------------------------------------------------

    def state_dim(self):
        return len(self.lanes)

    def action_dim(self):
        return len(self.green_phases)

    def encode_action(self, phase):
        onehot = np.zeros(len(self.green_phases), dtype=np.float32)
        onehot[phase] = 1.0
        return onehot

    # -- observation -------------------------------------------------------

    def state(self):
        return np.array(
            [traci.lane.getLastStepHaltingNumber(l) for l in self.lanes],
            dtype=np.float32,
        )

    # -- save / restore (for counterfactual branching) --------------------

    def save_state(self, path):
        traci.simulation.saveState(path)
        return {"path": path, "current_phase": self.current_phase}

    def load_state(self, snapshot):
        traci.simulation.loadState(snapshot["path"])
        self.current_phase = snapshot["current_phase"]

    # -- stepping ------------------------------------------------------

    def step(self, phase):
        """One control decision = STEP_LENGTH seconds, with a real yellow/all-red
        transition inserted if the requested phase differs from the current one."""
        if phase != self.current_phase:
            key = f"{self.current_phase}_{phase}"
            elapsed = 0
            if key in self.yellow_transitions:
                traci.trafficlight.setRedYellowGreenState(self.tl_id, self.yellow_transitions[key])
                for _ in range(YELLOW_LENGTH):
                    traci.simulationStep()
                    elapsed += 1
                if "g" in self.green_phases[self.current_phase]:
                    traci.trafficlight.setRedYellowGreenState(self.tl_id, self.yellow_transitions["all_red"])
                    for _ in range(CLEARANCE_LENGTH):
                        traci.simulationStep()
                        elapsed += 1
            traci.trafficlight.setRedYellowGreenState(self.tl_id, self.green_phases[phase])
            self.current_phase = phase
            for _ in range(STEP_LENGTH - elapsed):
                traci.simulationStep()
        else:
            for _ in range(STEP_LENGTH):
                traci.simulationStep()

        return self.state()


# -- controllers, same shapes as ctm_env's -----------------------------------

def controller_fixed_time(env, cycle):
    n = len(env.green_phases)

    def policy(t):
        return (t // cycle) % n
    return policy


def controller_random(env, switch_prob, rng=None):
    rng = rng if rng is not None else env.rng
    n = len(env.green_phases)
    phase = [env.current_phase]

    def policy(t):
        if rng.random() < switch_prob:
            phase[0] = int(rng.integers(0, n))
        return phase[0]
    return policy


def controller_max_pressure(env):
    def policy(t):
        state = env.state()
        pressures = []
        for idx, phase_state in env.green_phases.items():
            pressure = sum(
                state[i] for i, lane in enumerate(env.lanes)
                if i < len(phase_state) and phase_state[i] in ("G", "g")
            )
            pressures.append(pressure)
        return int(np.argmax(pressures))
    return policy
