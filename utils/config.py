from dataclasses import dataclass


@dataclass
class Config:
    ffn_depth: int = 2
    batch_size: int = 100
    epochs: int = 250
    early_stopping_patience: int = 20
    batch_normalization: bool = True
    learning_rate: float = 0.001
    label_type: int = 0
    hidden_dim = None
    validation_split = 0.25
