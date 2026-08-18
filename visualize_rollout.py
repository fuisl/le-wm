"""Generate the data behind the decoded-rollout visualization: for one
held-out SUMO episode, compute ground truth, teacher-forced probe decode,
and imagined-rollout probe decode, per approach, and dump to JSON for the
HTML artifact to render.

Mirrors eval_traffic.py's machinery (probe, latent_rollout) exactly - this
script only adds per-step logging and the approach-level aggregation needed
for the schematic.
"""

import json
from pathlib import Path

import numpy as np
import torch

from eval_traffic import apply_probe, fit_linear_probe, latent_rollout, load_model

SUMOCFG = "/home/fuisloy/cair/resco/resco_benchmark/environments/cologne1/cologne1.sumocfg"
DATA_DIR = "traffic_data_sumo"
RUN_NAME = "traffic-lewm-sumo"
WEIGHTS = "weights_epoch_60.pt"
HISTORY_SIZE = 3
EPISODE_IDX = 3          # fixed_time controller, val split
WINDOW_START = 41        # rising-congestion window found by inspection
LEAD_IN = 5              # pure ground-truth steps shown before the model's own context
HORIZON = 24              # imagined steps after context

# cologne1: 20 controlled links = 4 approaches x 5 links each (contiguous blocks,
# confirmed against traci.trafficlight.getControlledLanes ordering)
APPROACH_LABELS = ["Approach A", "Approach B", "Approach C", "Approach D"]
LINKS_PER_APPROACH = 5


def to_approach(vec20):
    """(20,) link-indexed vector -> (4,) approach-averaged vector."""
    return vec20.reshape(4, LINKS_PER_APPROACH).mean(axis=1)


def green_approaches(phase_state):
    """phase_state: 20-char SUMO signal string -> which of the 4 approaches has green."""
    out = []
    for a in range(4):
        block = phase_state[a * LINKS_PER_APPROACH: (a + 1) * LINKS_PER_APPROACH]
        out.append(any(c in ("G", "g") for c in block))
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(RUN_NAME, WEIGHTS, device)

    val_path = f"{DATA_DIR}/val.pt"
    W = fit_linear_probe(model, val_path, window=HISTORY_SIZE + 1, device=device)

    data = torch.load(val_path, weights_only=False)
    ep = data["episodes"][EPISODE_IDX]
    controller = ep["controller"]

    total_len = LEAD_IN + HISTORY_SIZE + HORIZON
    s0, s1 = WINDOW_START, WINDOW_START + total_len
    states = ep["state"][s0:s1]        # (total_len, 20)
    actions = ep["action"][s0:s1]      # (total_len, 8)
    assert states.shape[0] == total_len, "window runs past episode end"

    # teacher-forced: encode every true state independently (Embedder is pointwise
    # in time, so this needs no history) then decode through the probe
    with torch.no_grad():
        true_emb = model.encode({"state": torch.from_numpy(states).unsqueeze(0).to(device)})["emb"][0]
        teacher_forced_states = apply_probe(W, true_emb).cpu().numpy()  # (total_len, 20)

    # imagined rollout: real context (LEAD_IN : LEAD_IN+HISTORY_SIZE), then roll
    # forward autoregressively using the *true* action sequence, decode via probe
    ctx_start = LEAD_IN
    ctx_end = LEAD_IN + HISTORY_SIZE
    state_hist = torch.from_numpy(states[ctx_start:ctx_end])
    action_hist = torch.from_numpy(actions[ctx_start:ctx_end])
    action_future = torch.from_numpy(actions[ctx_end:])
    pred_emb = latent_rollout(model, state_hist, action_hist, action_future, HISTORY_SIZE, device)
    imagined_states = apply_probe(W, pred_emb).detach().cpu().numpy()  # (HORIZON, 20)

    # phase / green-approach bookkeeping, reading green_phases straight from SUMO
    from traffic.sumo_env import SumoTLEnv
    env = SumoTLEnv(SUMOCFG, seed=0)
    env.reset()
    green_phases = env.green_phases
    env.close()
    phase_idx_per_step = [int(np.argmax(a)) for a in actions]

    frames = []
    for t in range(total_len):
        is_context = ctx_start <= t < ctx_end
        is_imagined = t >= ctx_end
        gt_approach = to_approach(states[t]).tolist()
        tf_approach = to_approach(teacher_forced_states[t]).tolist()
        im_approach = to_approach(imagined_states[t - ctx_end]).tolist() if is_imagined else None
        phase_idx = phase_idx_per_step[t]
        frames.append({
            "t": t,
            "sim_time_s": (WINDOW_START + t) * 5,
            "region": "lead_in" if t < ctx_start else ("context" if is_context else "imagined"),
            "phase": phase_idx,
            "green": green_approaches(green_phases[phase_idx]),
            "ground_truth": gt_approach,
            "teacher_forced": tf_approach,
            "imagined": im_approach,
        })

    out = {
        "meta": {
            "scenario": "cologne1 (RESCO, real TAPAS-Cologne demand)",
            "controller": controller,
            "episode_idx": EPISODE_IDX,
            "window_start_step": WINDOW_START,
            "step_length_s": 5,
            "history_size": HISTORY_SIZE,
            "lead_in": LEAD_IN,
            "horizon": HORIZON,
            "approach_labels": APPROACH_LABELS,
            "checkpoint": f"{RUN_NAME}/{WEIGHTS}",
            "probe_train_mse": None,  # filled by fit_linear_probe's own print; see console
        },
        "frames": frames,
    }

    out_path = Path("traffic_data_sumo/rollout_viz.json")
    out_path.write_text(json.dumps(out))
    print(f"wrote {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
