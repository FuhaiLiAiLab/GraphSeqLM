# from transformers import pipeline
# protgpt2 = pipeline('text-generation', model="nferruz/ProtGPT2")
# # length is expressed in tokens, where each token has an average length of 4 amino acids.
# sequences = protgpt2("<|endoftext|>", max_length=100, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=10, eos_token_id=0)
# for seq in sequences:
#      print(seq)

import math
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Convert the sequence to a string like this
#(note we have to introduce new line characters every 60 amino acids,
#following the FASTA file format).

sequence = "<|endoftext|>\nMGEAMGLTQPAVSRAVARLEERVGIRIFNRTARAITLTDEGRRFYEAVAPLLAGIEMHGY\nRVNVEGVAQLLELYARDILAEGRLVQLLPEWAD\n<|endoftext|>"

# Load the model and tokenizer "nferruz/ProtGPT2"
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained("nferruz/ProtGPT2").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("nferruz/ProtGPT2")

# ppl function
def calculatePerplexity(sequence, model, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(sequence)).unsqueeze(0) 
    input_ids = input_ids.to(device)
    import pdb; pdb.set_trace()
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)

#And hence: 
ppl = calculatePerplexity(sequence, model, tokenizer)
print(ppl)