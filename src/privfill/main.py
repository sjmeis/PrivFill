import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from privfill.mechanisms import DPBart, DPPrompt

class PrivFill:
    def __init__(self, model_checkpoint, max_new_tokens=32, max_input_length=512, base_model=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_checkpoint = model_checkpoint
        self.max_new_tokens = max_new_tokens
        self.max_input_length = max_input_length
        self.base_model = base_model

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint).to(self.device)

    def privatize(self, text):
        sentences = nltk.sent_tokenize(text)
        replace = []
        for s in sentences:
            temp = text.replace(s, "[blank]")
            inputs = [temp]
            inputs = self.tokenizer(inputs, max_length=self.max_input_length, truncation=True, return_tensors="pt").input_ids.to(self.device)
            output = self.model.generate(inputs, min_new_tokens=5, do_sample=True, max_new_tokens=self.max_new_tokens, pad_token_id=50256)
            decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True).replace(temp, "")
            
            if self.base_model is None:
                replace.append(decoded_output)
            else:
                replace.append(nltk.sent_tokenize(decoded_output.strip())[0])
        return " ".join(replace)
    

class PrivFillDPBart:
    def __init__(self, model_checkpoint, max_new_tokens=32, max_input_length=512):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_checkpoint = model_checkpoint
        self.max_new_tokens = max_new_tokens
        self.max_input_length = max_input_length

        self.model = DPBart(model=model_checkpoint)

    def privatize(self, text, epsilon):
        sentences = nltk.sent_tokenize(text)
        eps = epsilon / len(sentences)
        inputs = []
        for s in sentences:
            temp = text.replace(s, "[blank]")
            inputs.append(temp)
        
        return self.model.privatize_batch(inputs, epsilon=eps)
    

class PrivFillDP:
    def __init__(self, model_checkpoint, max_new_tokens=32, max_input_length=512):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_checkpoint = model_checkpoint
        self.max_new_tokens = max_new_tokens
        self.max_input_length = max_input_length

        self.model = DPPrompt(model_checkpoint=model_checkpoint)

    def privatize(self, text, epsilon):
        sentences = nltk.sent_tokenize(text)
        inputs = []
        for s in sentences:
            temp = text.replace(s, "[blank]")
            inputs.append(temp)

        output = self.model.privatize_dp(inputs, epsilon)
        return " ".join(output)