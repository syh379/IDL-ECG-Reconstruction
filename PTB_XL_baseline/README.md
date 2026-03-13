# ECG Baseline Replication + Lead Masking Experiments with xresnet1d101 on PTB-XL

## Attribution

This code directory is based on the original benchmarking framework from [Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL](https://doi.org/10.1109/jbhi.2020.3022989) by Strodthoff et al., which builds on the [PTB-XL dataset](https://www.nature.com/articles/s41597-020-0495-6). The original code and framework can be found at the [authors' repository](https://github.com/helme/ecg_ptbxl_benchmarking).

## Overview

We have replicated the baseline established in the original paper and adapted the original scripts to run our own experiments investigating the effect of ECG lead masking on classification performance using the **xresnet1d101** model. Specifically, the experiments explore how well the model performs when trained and evaluated on subsets of ECG leads.

## Output

All experiment results are stored in the `output/` folder:

| Folder | Description |
|--------|-------------|
| `exp_12_lead_baseline` | Full 12-lead baseline using xresnet1d101 |
| `exp_leads_I` | Masking experiment using only lead I |
| `exp_leads_II` | Masking experiment using only lead II |
| `exp_leads_III` | Masking experiment using only lead III |
| `exp_leads_I_II` | Masking experiment using leads I and II |
| `exp_leads_I_II_III` | Masking experiment using leads I, II, and III |

Each experiment folder contains trained model checkpoints and results (`models/`), evaluation data (`data/`), and bootstrap sample IDs used for evaluation.

## References

```bibtex
@article{Strodthoff:2020Deep,
doi = {10.1109/jbhi.2020.3022989},
url = {https://doi.org/10.1109/jbhi.2020.3022989},
year = {2021},
volume={25},
number={5},
pages={1519-1528},
publisher = {Institute of Electrical and Electronics Engineers ({IEEE})},
author = {Nils Strodthoff and Patrick Wagner and Tobias Schaeffter and Wojciech Samek},
title = {Deep Learning for {ECG} Analysis: Benchmarks and Insights from {PTB}-{XL}},
journal = {{IEEE} Journal of Biomedical and Health Informatics}
}

@article{Wagner:2020PTBXL,
doi = {10.1038/s41597-020-0495-6},
url = {https://doi.org/10.1038/s41597-020-0495-6},
year = {2020},
publisher = {Springer Science and Business Media {LLC}},
volume = {7},
number = {1},
pages = {154},
author = {Patrick Wagner and Nils Strodthoff and Ralf-Dieter Bousseljot and Dieter Kreiseler and Fatima I. Lunze and Wojciech Samek and Tobias Schaeffter},
title = {{PTB}-{XL},  a large publicly available electrocardiography dataset},
journal = {Scientific Data}
}

@misc{Wagner2020:ptbxlphysionet,
title={{PTB-XL, a large publicly available electrocardiography dataset}},
author={Patrick Wagner and Nils Strodthoff and Ralf-Dieter Bousseljot and Wojciech Samek and Tobias Schaeffter},
doi={10.13026/qgmg-0d46},
year={2020},
journal={PhysioNet}
}

@article{Goldberger2020:physionet,
author = {Ary L. Goldberger  and Luis A. N. Amaral  and Leon Glass  and Jeffrey M. Hausdorff  and Plamen Ch. Ivanov  and Roger G. Mark  and Joseph E. Mietus  and George B. Moody  and Chung-Kang Peng  and H. Eugene Stanley },
title = {{PhysioBank, PhysioToolkit, and PhysioNet}},
journal = {Circulation},
volume = {101},
number = {23},
pages = {e215-e220},
year = {2000},
doi = {10.1161/01.CIR.101.23.e215}
}
```