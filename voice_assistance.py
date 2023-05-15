from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# HYPERPARAMETERS and CONSTANTS
commands = [
    "What is in front of me?",
    "Is there a car?",
    "Is there a dog?",
    "Am I walking on the left lane?",
    "Has the bus arrived?",
    "Is it safe to move?",
    "Turn on Flash",
    "Turn off Flash",
]

commandTensors = []

MAX_LENGTH = 128

model_name = 'sentence-transformers/bert-base-nli-mean-tokens'

# LOAD MODEL
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def getTensor(sentence) :
    tokens = {'input_ids' : [], 'attention_mask' : []}

    # Tokenize the sentences
    new_tokens = tokenizer.encode_plus(sentence, max_length=MAX_LENGTH, truncation=True, padding='max_length', return_tensors = 'pt')
    tokens['input_ids'].append(new_tokens['input_ids'][0])
    tokens['attention_mask'].append(new_tokens['attention_mask'][0])

    # Note : the shape of tokens is ([1,128]), 128 is the number of tokens
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

    # Send the tokens to BERT Model
    outputs = model(**tokens)
    # dimension is ([1,128,768])

    # We extract the last tensor - Contains all the semantic information
    embeddings = outputs.last_hidden_state

    # Mean Pooling
    #Only take the contribution of true tokens and 0 contribution for padding tokens
    attention = tokens['attention_mask']

    # To match the dimensions for valid matrix multiplication
    mask = attention.unsqueeze(-1).expand(embeddings.shape).float()
    mask_embeddings = embeddings * mask

    # Take sum of tensor values corresponding each token
    summed = torch.sum(mask_embeddings, 1)

    # The point of clamping is to prevent any divide by zero errors
    # All zeroes will be considered as 1e-9
    counts = torch.clamp(mask.sum(1), min = 1e-9)

    mean_pooled = summed / counts
    mean_pooled = mean_pooled.detach().numpy()
    
    return mean_pooled


querySentence = 'I am scared of animals'
def func():
    for sentence in commands:
        commandTensors.append(getTensor(sentence))
    
    querySentenceTensor = getTensor(querySentence)

    cosineSimilarity = []

    for tensor in commandTensors:
        cosineSimilarity.append(cosine_similarity(querySentenceTensor,tensor)[0])

    print(cosineSimilarity)
    nparray = np.array(cosineSimilarity)
    index = np.argmax(nparray)

    print(commands[index], nparray[index])
func()