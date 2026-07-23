try:
    from src.deepseek.client import DeepSeekClient
    _deepseek_ok = True
except ImportError:
    DeepSeekClient = None
    _deepseek_ok = False

from src.deepseek.prompt_builder import (
    build_arc_prompt,
    SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
)
from src.deepseek.output_parser import parse_output_grid, grids_match, evaluate_predictions, parse_from_json_mode

__all__ = [
    "DeepSeekClient",
    "build_arc_prompt",
    "parse_output_grid",
    "parse_from_json_mode",
    "grids_match",
    "evaluate_predictions",
    "SYSTEM_PROMPT",
    "FEW_SHOT_EXAMPLES",
]
