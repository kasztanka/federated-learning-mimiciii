from dataclasses import dataclass


@dataclass
class Config:
    ffn_depth: int = 2
    batch_size: int = 100
    epochs: int = 250
    early_stopping_patience: int = 20
    batch_normalization: bool = True
    learning_rate: float = 0.001
    loss_func: str = 'binary_crossentropy'
    final_activation: str = 'sigmoid'
    label_type: int = 0
