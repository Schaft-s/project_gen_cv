from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
NOISE_BANKS = DATA / "noise_banks"
REFERENCES = DATA / "references"


def load_default_config() -> dict:
    import yaml

    path = CONFIGS / "default.yaml"
    with path.open() as f:
        return yaml.safe_load(f)
