try:
    from src.varc.model import VARCConfig, VARCModel
    from src.varc.train import train_epoch, train_model, ttt_finetune, save_checkpoint, load_checkpoint
    from src.varc.data import ARCDataset, task_to_canvas, DEFAULT_CANVAS_SIZE, BG_CLASS, BD_CLASS
    from src.varc.evaluate import run_inference, predict_grid, evaluate_on_tasks, compute_accuracy
    _varc_ok = True
except ImportError:
    VARCConfig = None
    VARCModel = None
    train_epoch = None
    train_model = None
    ttt_finetune = None
    save_checkpoint = None
    load_checkpoint = None
    ARCDataset = None
    task_to_canvas = None
    DEFAULT_CANVAS_SIZE = 64
    BG_CLASS = 10
    BD_CLASS = 11
    run_inference = None
    predict_grid = None
    evaluate_on_tasks = None
    compute_accuracy = None
    _varc_ok = False

__all__ = [
    "VARCConfig",
    "VARCModel",
    "train_epoch",
    "train_model",
    "ttt_finetune",
    "save_checkpoint",
    "load_checkpoint",
    "ARCDataset",
    "task_to_canvas",
    "DEFAULT_CANVAS_SIZE",
    "BG_CLASS",
    "BD_CLASS",
    "run_inference",
    "predict_grid",
    "evaluate_on_tasks",
    "compute_accuracy",
]
