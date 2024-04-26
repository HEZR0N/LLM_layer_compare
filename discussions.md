#### Consistency check method between these layers for factuality analysis
Some research papers have proposed using different models several times or the same model several times to see if the same repsonse can be generated mulitple times. If so, then the response is believed to be more likely true.    
I believe the same could be true of comparing the different if applied in a certain way. I believe the constrains would have to be loose though, seeing aa in the results I obtained, the correct resposne did not appear 
in the top 32 highest probability tokens until layer 23. Here's how I would use consistncy to check for factuality
 - The high the rank a token has (ie its probabilty in the probabilty distribution after the soft_max; how likely the model thought the token was), the more likely the token is to be factual.
    - For the token was in the layer (maybe add a limit like the top 100 most likely tokens in that layer), add to the `factuality_probability` score the weight of that rank. The highest ranks (1, 2, 3) have higher weights than lower ranks (98, 99, 100) 
 - If a token/token sequence was in a layer, it is more likely to be factual.
    - For each layer the token/token sequence was in, add to the `factuality_probability` score the weight of that layer. The last layers have the highest weights and thus the most influence on the score
I would train a simple model to determine the most appropiate weights to assign:

```
factuality_probability = 0
RANK_WEIGHTS = get_rank_weights()

for layer in layers:
  cur_tokens = layer['tokens'] 
  if token in cur_tokens:
    factuality_probability += layer['weight'] * RANK_WEIGHTS[cur_tokens.index(token) + 1]
```

`factuality_probability = in_layer1 * layer1_weight * rank_N_weight + ... + in_layer32 * layer32_weight * rank_N_weight`

#### Layers effect on diffeent metrics
Becasue I did not use the same dataset that was used for finetuning to perform layer analysis, I.

I know the purpose of comparing metrics is to see which quality improved first: 
 - Longest common sequence of words (rouge)
 - Number of common n-grams (bleu)
 - Semantics/embeddings similarity

Because bleu and rouge are strictly word based (not considering semantics), 
they remained 0 the first 26 layers and became 1 for layer 27 - 32 (as the correct word had not been chosen for the highest rank until layer 27 and remained that way).
I if were to consider the top 32 tokens in the metrics instad of just the token with the highest probability, 
the score change from 0 to 1 is slightly more gradual, but not by much (because the correct word still doesn't show up in any of top 32 words until layer 23).
The bert score is a lot more helpful, especially when considering the top 32 tokens in the metrics instad of just the token with the highest probability, as it shows when the model starts to predict at least plausible words, 
which start showing up as early as layer 3, and by the final layers, most of the words in the top 32 tokens have similar smeantics and could have been considered the correct next word.
