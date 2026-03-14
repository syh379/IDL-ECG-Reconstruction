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
@misc{lence2025ecgrecoverdeeplearningapproach,
      title={ECGrecover: a Deep Learning Approach for Electrocardiogram Signal Completion}, 
      author={Alex Lence and Federica Granese and Ahmad Fall and Blaise Hanczar and Joe-Elie Salem and Jean-Daniel Zucker and Edi Prifti},
      year={2025},
      eprint={2406.16901},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2406.16901}, 
}
```
