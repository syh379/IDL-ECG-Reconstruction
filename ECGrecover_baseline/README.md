# ECGrecover Reconstruction Baseline

## Attribution

This code directory contains our adaptation of [ECGrecover: A Deep Learning Approach for Electrocardiogram
Signal Completion](https://arxiv.org/pdf/2406.16901). The original model and code were developed by Lence et al. and have been adapted here to evaluate baseline reconstruction performance on the PTB-XL dataset. The original code can be found at [the author's repository](https://github.com/UMMISCO/ecgrecover).

## Overview

We adopted the original ECGrecover repository and wrote a separate Jupyter notebook (ECGrecover_baseline_PTB_XL.ipynb) to apply the trained ECGrecover model to the **PTB-XL** dataset, in order to replicate and evaluate baseline reconstruction metrics.

## Output
The model and the processed data used for evaluation are in the `model.zip` file. The detailed metrics produced are in the `results.zip` file due to size limit.

## References

```bibtex
@inproceedings{lence2025ecgrecover,
  title={{ECGrecover}: A Deep Learning Approach for Electrocardiogram Signal Completion},
  author={Lence, Alex and Granese, Federica and Fall, Ahmad and Hanczar, Blaise and Salem, Joe-Elie and Zucker, Jean-Daniel and Prifti, Edi},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2025},
  organization={ACM},
  note={arXiv preprint arXiv:2406.16901}
}
```
