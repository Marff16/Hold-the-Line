# Hold The Line

Hold The Line, or HDL, is a fixed-map MVP for a continuous 2D multi-agent
reinforcement learning environment where blue defender drones detect, track,
and abstractly intercept red scout drones before the scouts collect enough
information about scouting objectives. The README.md will be updated further in later commits.

## Setup

Requires [Poetry](https://python-poetry.org/). If it's not installed yet:

```bash
pip install poetry
```

Then install the project's dependencies:

```bash
poetry install
```

## Run the web simulator

```bash
poetry run uvicorn src.server.app:app --reload
```

Open http://127.0.0.1:8000 in a browser. Use the Instance dropdown to switch
between the scenario files in `instances/*.json`.

## Run the tests

```bash
poetry run python -m unittest discover -s tests
```

## Run an example script

```bash
poetry run python examples/random_rollout.py
poetry run python examples/render_snapshot.py
poetry run python examples/generate_maps.py
```
