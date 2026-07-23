from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARC_AGI1_DIR = DATA_DIR / "ARC-AGI"
ARC_AGI2_DIR = DATA_DIR / "ARC-AGI-2"
NSA_DIR = DATA_DIR / "NSA"

ARC_AGI1_TRAINING_DIR = ARC_AGI1_DIR / "data" / "training"
ARC_AGI1_EVALUATION_DIR = ARC_AGI1_DIR / "data" / "evaluation"

RESULTS_DIR = PROJECT_ROOT / "results"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
