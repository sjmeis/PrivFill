import pandas as pd
import random
import nltk
import re
import numpy as np
from tqdm.auto import tqdm

import string
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

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

        if base_model is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, add_bos_token=True, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.base_model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config)
            self.model = PeftModel.from_pretrained(base_model, model_checkpoint)
        else:
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