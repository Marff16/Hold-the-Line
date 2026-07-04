"""Generate and print valid procedural map candidates."""

from __future__ import annotations

from src.map_generator import describe_map, generate_valid_map, validate_map


def main() -> None:
    for seed in range(3):
        config = generate_valid_map(seed=seed, building_count=7)
        result = validate_map(config)
        print(f"\n=== valid map seed={seed} routes={result.approach_routes} ===")
        print(describe_map(config))


if __name__ == "__main__":
    main()
