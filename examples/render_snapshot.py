"""Save visual renders of fixed and generated Hold The Line maps."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hold_the_line_matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from src import generate_valid_map, parallel_env


def save_snapshot(path: Path, *, seed: int, generated_map_seed: int | None = None) -> None:
    map_config = generate_valid_map(seed=generated_map_seed) if generated_map_seed is not None else None
    env = parallel_env(map_config=map_config, render_mode="rgb_array")
    env.reset(seed=seed)

    if generated_map_seed is not None:
        rng = np.random.default_rng(seed)
        for _ in range(18):
            actions = {agent: rng.uniform(-1.0, 1.0, size=2).astype(np.float32) for agent in env.agents}
            env.step(actions)
            if not env.agents:
                break

    image = env.render()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, image)
    env.close()


def main() -> None:
    save_snapshot(Path("renders/fixed_map.png"), seed=42)
    save_snapshot(Path("renders/generated_map_seed_0.png"), seed=42, generated_map_seed=0)
    print("saved renders/fixed_map.png")
    print("saved renders/generated_map_seed_0.png")


if __name__ == "__main__":
    main()
