import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from experiments.scp_experiment import SCP_Experiment
from configs.fastai_configs import conf_fastai_xresnet1d101

if __name__ == '__main__':
    datafolder = '../data/ptbxl/'
    outputfolder = '../output/'

    models = [conf_fastai_xresnet1d101]

    e = SCP_Experiment('exp1.1.1', 'superdiagnostic', datafolder, outputfolder, models)
    e.prepare()
    e.perform()
    e.evaluate()