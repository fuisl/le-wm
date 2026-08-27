"""Thin wandb wrapper: online if credentials exist, offline otherwise (runs land
in ./wandb/offline-run-* and can be pushed later with `wandb sync`), or a no-op
stub if logging wasn't requested. Keeps the training / rollout scripts from
caring which case they're in.
"""

import os


class _NoOp:
    def log(self, *a, **k):
        pass

    def finish(self, *a, **k):
        pass

    def __bool__(self):
        return False


def _has_creds():
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        import netrc
        n = netrc.netrc(os.path.expanduser("~/.netrc"))
        return any("wandb.ai" in h for h in n.hosts)
    except Exception:
        return False


def init(enabled, project, name, config, group=None):
    if not enabled:
        return _NoOp()
    import wandb
    mode = "online" if _has_creds() else "offline"
    if mode == "offline":
        print(f"[wandb] no credentials found -> offline mode; "
              f"`wandb login` then `wandb sync wandb/offline-run-*` to push")
    run = wandb.init(project=project, name=name, config=config, group=group,
                     mode=mode, reinit=True)
    return run
