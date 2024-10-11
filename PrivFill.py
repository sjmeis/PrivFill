import pandas as pd
import nltk
import numpy as np
from tqdm.auto import tqdm

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

import LLMDP

class PrivFill():
    base_model = None
    model_checkpoint = None
    max_new_tokens = None
    max_input_length = None
    model = None
    tokenizer = None
    device = None

    def __init__(self, model_checkpoint, max_new_tokens=32, max_input_length=512, base_model=None):
        if torch.cuda.is_available() == True:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model_checkpoint = model_checkpoint
        self.max_new_tokens = max_new_tokens
        self.max_input_length =  max_input_length

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint).to("cuda")

    def privatize(self, text):
        sentences = nltk.sent_tokenize(text)
        replace = []
        for s in sentences:
            temp = text.replace(s, "[blank]")
            inputs = [temp]
            inputs = self.tokenizer(inputs, max_length=self.max_input_length, truncation=True, return_tensors="pt").input_ids.to("cuda")
            output = self.model.generate(inputs, min_new_tokens=5, do_sample=True, max_new_tokens=self.max_new_tokens, pad_token_id=50256)
            decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True).replace(temp, "")
            if self.base_model is None:
                replace.append(decoded_output)
            else:
                replace.append(nltk.sent_tokenize(decoded_output.strip())[0])
        return " ".join(replace)
    
class PrivFillDPBart():
    base_model = None
    model_checkpoint = None
    max_new_tokens = None
    max_input_length = None
    model = None
    tokenizer = None
    device = None

    def __init__(self, model_checkpoint, max_new_tokens=32, max_input_length=512):
        if torch.cuda.is_available() == True:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model_checkpoint = model_checkpoint
        self.max_new_tokens = max_new_tokens
        self.max_input_length =  max_input_length

        self.model = LLMDP.DPBart(model=model_checkpoint)

    def privatize(self, text, epsilon):
        sentences = nltk.sent_tokenize(text)
        eps = epsilon / len(sentences)
        inputs = []
        for s in sentences:
            temp = text.replace(s, "[blank]")
            inputs.append(temp)
        
        output = self.model.privatize_batch(inputs, epsilon=eps)

        return output
    
class PrivFillDP():
    base_model = None
    model_checkpoint = None
    max_new_tokens = None
    max_input_length = None
    model = None
    tokenizer = None
    device = None

    def __init__(self, model_checkpoint, max_new_tokens=32, max_input_length=512):
        if torch.cuda.is_available() == True:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model_checkpoint = model_checkpoint
        self.max_new_tokens = max_new_tokens
        self.max_input_length =  max_input_length

        self.model = LLMDP.DPPrompt(model_checkpoint=model_checkpoint)

    def privatize(self, text, epsilon):
        sentences = nltk.sent_tokenize(text)
        inputs = []
        for s in sentences:
            temp = text.replace(s, "[blank]")
            inputs.append(temp)

        output = self.model.privatize_dp(inputs, epsilon)
        return " ".join(output)