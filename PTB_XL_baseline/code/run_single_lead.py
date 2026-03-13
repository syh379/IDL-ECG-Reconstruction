import os
import numpy as np
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from experiments.scp_experiment import SCP_Experiment
from configs.fastai_configs import conf_fastai_xresnet1d101

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def mask_leads(X, keep_leads):
    """Zero out all leads except the ones in keep_leads."""
    X_masked = np.zeros_like(X)
    for lead in keep_leads:
        X_masked[:, :, lead] = X[:, :, lead]
    return X_masked


if __name__ == '__main__':
    datafolder = '../data/ptbxl/'
    outputfolder = '../output/'

    models = [conf_fastai_xresnet1d101]

    # Lead combinations to test
    lead_sets = [
        [1],        # Lead II
        [2],        # Lead III
        [0,1],      # Lead I + II
        [0,1,2]     # Lead I + II + III
    ]

    for leads in lead_sets:

        lead_names = "_".join([LEAD_NAMES[l] for l in leads])
        exp_name = f'exp_leads_{lead_names}'

        print(f'\nRunning experiment with leads: {lead_names}')

        e = SCP_Experiment(exp_name, 'superdiagnostic', datafolder, outputfolder, models)
        e.prepare()

        print(f'Original X_train shape: {e.X_train.shape}')

        e.X_train = mask_leads(e.X_train, leads)
        e.X_val   = mask_leads(e.X_val, leads)
        e.X_test  = mask_leads(e.X_test, leads)

        print(f'Masked X_train shape: {e.X_train.shape} (active leads: {lead_names})')

        e.perform()
        e.evaluate()

        print(f'\nResults saved to {outputfolder}{exp_name}/models/')