# import libraries
import os
from huggingface_hub.hf_api import HfFolder
import torch
from torch.nn import functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import nltk
from nltk.translate.bleu_score import sentence_bleu
nltk.download('punkt')
from rouge import Rouge
from bert_score import BERTScorer
from evaluate import load
import matplotlib.pyplot as plt

# load model and tokenizer
access_token="YOUR_HF_TOKEN"
model_path = "model/path"
model = AutoModelForCausalLM.from_pretrained(model_path, token=access_token)

print(model)

# initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, token=access_token)
tokenizer.pad_token = tokenizer.eos_token

# load dataset
#dataset = load_dataset("lighteval/mmlu", "abstract_algebra", split="validation")
#dataset = dataset.select(range(10))

# Example prompts
#prompt=f'Answer this question with one word: Is the sky blue?'
#prompt=f'The Question is {dataset["question"][0]}. You will only answer the question with one of the options provided in this list: {dataset["choices"][0]}
data = [["Birds fly high in the ", "sky"], ["The color of the sky is ", "blue"], ["The color of grass is ", "green"], ["People drive in ", "cars"]]

# Generate outputs for each data point. 10?
outputs = []
for i in range(len(data)):
  with torch.no_grad():
    tokens = tokenizer(data[i][0], return_tensors='pt', padding=True, max_length=1024)
    output = model.generate(**tokens, num_return_sequences=1, return_dict_in_generate=True, output_logits=True, output_hidden_states=True, max_new_tokens=1)
    outputs.append(output)
    
# Get tokens and scores (at least top 20) for layers 8, 16, 24, and 32 by default
def top_best_tokens(output, layer_num, token_limit=20):
  # Get the probabilty distribution for the output (ie what's the next most likely token)
  token_probs = F.softmax(model.lm_head(output.hidden_states[0][layer_num][0,-1, :]), dim=-1)
  # print(f"The number of tokens in the vocabulary is {len(token_probs)}")
  # ^I had printed to confirm that there are 32000 tokens in the vocabulary
  token_prob_list = token_probs.tolist()

  # Sort the list so we can look at the tokens with the highest probabilties
  best_token_probs = sorted(token_prob_list, reverse=True)[:token_limit]

  # Create a dictionary to map a given probability to it's token
  prob_to_token_dict = {}
  for i in range(len(token_prob_list)):
    prob_to_token_dict[token_prob_list[i]]=i

  print("Layer ", layer_num, " best prob: ", best_token_probs[0], ", best token: ", tokenizer.decode(prob_to_token_dict[best_token_probs[0]], skip_special_tokens=True))

  # Create of list the best tokens from the best token probabilities
  best_decoded_tokens = []
  for i in range(token_limit):
    best_decoded_tokens.append(tokenizer.decode(prob_to_token_dict[best_token_probs[i]], skip_special_tokens=True))
  return best_decoded_tokens, best_token_probs
  
  
def get_layer_tokens_and_probs(output, layer_nums=[7, 15, 23, 31], token_limit=20, plot=True):
  layer_to_best_tokens_and_probs = {}
  # Plot tokens and scores
  for i in layer_nums:
    best_decoded_tokens, best_token_probs = top_best_tokens(output, i, token_limit)
    # Plot the decoded tokens and their probabilities
    if plot:
      plt.bar(best_decoded_tokens, best_token_probs)
      plt.title(f'Layer {i+1}: {token_limit} Most Likely Tokens')
      plt.xlabel('Best Predctions')
      plt.xticks(rotation=45)
      plt.ylabel('Scores')
      plt.show()
    layer_to_best_tokens_and_probs[i]=[best_decoded_tokens, best_token_probs]
  return layer_to_best_tokens_and_probs
  
# This will plot the token layers' probabilities for the first example
compare_layer_probs_of_first_output = get_layer_tokens_and_probs(outputs[0])

# Get top 32 tokens for all 32 layers
tab="\t"
compare_layer_probs_output = get_layer_tokens_and_probs(outputs[0], [i for i in range(32)], 32, False)
print(f'Layer N: {tab.join([str(i+1) for i in range(32)])}')
for i in compare_layer_probs_output.keys():
  print(f'Layer {i+1}: {tab.join(compare_layer_probs_output[i][0])}')
  
# Evaluation Metrics
roug = Rouge()
bertscore = load("bertscore")

def get_metrics(new_response, ground_truth):
  blue = nltk.translate.bleu_score.sentence_bleu([ground_truth.split()], new_response.split())
  red = roug.get_scores(new_response, ground_truth)
  red = red[0]['rouge-l']['f']
  raw_bert = bertscore.compute(predictions=[new_response], references=[ground_truth], model_type="distilbert-base-uncased")
  bert = raw_bert['f1'][0]
  return blue, red, bert

print("Metrics for: 'Birds fly high in the... sky'")
print("Layer N: \t\tBLEU\t\tRouge\t\tBERT")
for layer_num in compare_layer_probs_of_first_output.keys():
  # print(compare_layer_probs_of_first_output[layer_num][0][0])
  BLEU, Roug, BERTScore = get_metrics(compare_layer_probs_of_first_output[layer_num][0][0], data[0][1])
  print(f"Layer {layer_num+1}: \t\t{BLEU:.4f}\t\t{Roug:.4f}\t\t{BERTScore:.4f}")
