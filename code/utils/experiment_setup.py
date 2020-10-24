from dataclasses import dataclass


@dataclass
class ExperimentSetup:
    output_size: int
    folds_file: str
    y_label: str
    results_filename: str


MORTALITY_SETUP = ExperimentSetup(1, 'folds_ep_mor', 'adm_labels_all', 'mortality_binary_classification')
ICD9_SETUP = ExperimentSetup(20, 'folds_ep_icd9_multi', 'y_icd9', 'icd9_multilabel_classification')
