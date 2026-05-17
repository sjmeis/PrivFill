<div align="center">

  # PrivFill

  [![PyPI version](https://img.shields.io/pypi/v/privfill.svg)](https://pypi.org/project/privfill/)
  [![GitHub stars](https://img.shields.io/github/stars/sjmeis/PrivFill.svg?style=social)](https://github.com/sjmeis/PrivFill/stargazers)
  [![License](https://img.shields.io/github/license/sjmeis/PrivFill.svg)](https://github.com/sjmeis/PrivFill/blob/main/LICENSE)

</div>

`PrivFill` is a Python package providing LLM-based local Differential Privacy (DP) mechanisms for text privatization via sentece infilling. It offers easy-to-use wrappers for fine-tuned Hugging Face models.
This software was originally presented in the NAACL 2025 findings paper: *On the Impact of Noise in Differentially Private Text Rewriting*

## Installation

Install the package locally in editable mode from your project's root directory:

```bash
pip install privfill
```

### Core Prerequisites:

- Python $\geq$ 3.9
- PyTorch (CUDA recommended for faster inference)
- Transformers & NLTK

## Basic Usage & Model Selection
Instead of typing Hugging Face repository paths, you can choose from the three built-in models using the `SupportedModels` enum.

```python
import privfill

# Choose between FLAN_T5_BASE, FLAN_T5_LARGE, and BART_LARGE
engine = privfill.load_pipeline(privfill.SupportedModels.FLAN_T5_BASE, DP=True)

text = "This is a long private document ... which contains sensitive information and should be privatized,"
private_text = engine.privatize(text, epsilon=10)

print(private_text)
```

As described in the paper, we also create an analagous, non-DP variant of `PrivFill`. The usage is very similar:

```python
engine = privfill.load_pipeline(privfill.SupportedModels.FLAN_T5_BASE, DP=False)
private_text = engine.privatize(text)
```

### Available Models

| Enum                 | Hugging Face Repository              | Base Mechanism               |
|-------------------------------|--------------------------------------|-------------------------|
| SupportedModels.FLAN_T5_BASE  | sjmeis/flan-t5-base-infill-combined  | DP-Prompt   |
| SupportedModels.FLAN_T5_LARGE | sjmeis/flan-t5-large-infill-combined | DP-Prompt       |
| SupportedModels.BART_LARGE    | sjmeis/bart-large-infill-combined    | DP-BART |


## Comparison Code
We also include the LLMDP class code for `DP-BART` and `DP-Prompt`, as used in the paper.

```python
X = LLMDP.DPPrompt()
# or
X = LLMDP.DPBart()

# then
X.privatize(text, epsilon)
```

## Citation
If you use or extend `PrivFill`, please consider citing the original work:

```
@inproceedings{meisenbacher-etal-2025-impact,
    title = "On the Impact of Noise in Differentially Private Text Rewriting",
    author = "Meisenbacher, Stephen  and
      Chevli, Maulik  and
      Matthes, Florian",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2025",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-naacl.32/",
    doi = "10.18653/v1/2025.findings-naacl.32",
    pages = "514--532",
    ISBN = "979-8-89176-195-7"
}
```
