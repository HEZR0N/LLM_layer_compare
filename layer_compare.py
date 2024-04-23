# import libraries
import os
from huggingface_hub.hf_api import HfFolder
import torch
from torch.nn import functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

import nltk
from rouge import Rouge
from bert_score import BERTScorer
from evaluate import load

# load model and tokenizer

model_path = "./mistral-finetuned"

model = AutoModelForCausalLM.from_pretrained(model_path)

print(model)

tokenizer = AutoTokenizer.from_pretrained(model_path)

print('done')

# load dataset

dataset = load_dataset("lighteval/mmlu", "abstract_algebra", split="validation")

dataset = dataset.select(range(10))

# Example prompt
#prompt=f'Answer this question with one word: Is the sky blue?'
#prompt='The color of the sky is'
#prompt='Answer this question in one word: Is grass red, blue, or green?'
prompt="Birds fly high up in the "
########prompt=f'The Question is {dataset["question"][0]}. You will only answer the question with one of the options provided in this list: {dataset["choices"][0]}'
print(prompt)


# Generate outputs for each data point. 10?
with torch.no_grad():
	tokens = tokenizer(prompt, return_tensors='pt', padding=True, max_length=1024)
	#output = model.generate(**tokens, output_logits=True, output_hidden_states=True, max_new_tokens=5)
	output = model.generate(**tokens, num_return_sequences=1, return_dict_in_generate=True, output_logits=True, output_hidden_states=True, max_new_tokens=1)


#tokens['decoder_input_ids'] = tokens['input_ids'].clone()
#print(tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True))
print(tokenizer.batch_decode(output.sequences[:, tokens['input_ids'].shape[-1]:], skip_special_tokens=True))

#answer_start_index = output.start_logits.argmax()
#answer_end_index = output.end_logits.argmax()
#predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
#print(tokenizer.decode(predict_answer_tokens))

#print(tokenizer.decode(output[0], skip_special_tokens=True))

#decoded_outputs = [tokenizer.decode(output_sequences[0], skip_special_tokens=True) for output_sequences in output.sequences]
#print(decoded_outputs)

# Get tokens and scores (at least top 5) for layers 8, 16, 24, and 32

with torch.no_grad():
	prob = F.softmax(model.lm_head(output.hidden_states[0][9][0,-1, :]), dim=-1)
	print(prob)

# Plot tokens and scores


# Evaluate

