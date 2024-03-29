# PrivFill

Anonymous repository containing software for the COLM 2024 submission: *No Noise Needed: Language Model Sentence Infilling as an Alternative to Differentially Private Text Rewriting*

## Basic Usage ## 
`X = PrivFill.PrivFill(model_checkpoint="/path/to/infilling_model")`

`X.privatize(text)`

## Models ##
We make our three sentence infilling models public. They can be found at this [link](https://drive.google.com/drive/folders/12m1av9PY1X7S-cwd9y_8nepBPMtVju0C?usp=sharing).

## Comparison Code ##
We also includ the LLMDP class code for `DP-BART` and `DP-Prompt`, as used in the paper.

`X = LLMDP.DPPrompt()` or `X = LLMDP.DPBart()`

`X.privatize(text, epsilon)`
