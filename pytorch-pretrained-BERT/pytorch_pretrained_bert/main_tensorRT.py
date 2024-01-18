import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
from torch2trt import torch2trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenized input
text = "Who was Jim Henson ? Jim Henson was a puppeteer"
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 6
tokenized_text[masked_index] = '[MASK]'
assert tokenized_text == ['who', 'was', 'jim', 'henson', '?', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer']

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

# Predict all tokens
predictions = model(tokens_tensor, segments_tensors)

# confirm we were able to predict 'henson'
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token)
assert predicted_token == 'henson'

# Convert PyTorch model to TensorRT engine
model_trt = torch2trt(model, [tokens_tensor, segments_tensors])

# Allocate device memory and copy input data to device
d_input = cuda.mem_alloc(1 * tokens_tensor.element_size())
d_output = cuda.mem_alloc(1 * predictions.element_size())

# Transfer input data to device
cuda.memcpy_htod(d_input, tokens_tensor.numpy().astype(np.float32))

# Run inference
context = model_trt.create_execution_context()
context.execute(1, [int(d_input), int(d_output)])

# Transfer output data to host and post-process
cuda.memcpy_dtoh(predictions, d_output)

# Confirm prediction
predicted_index = np.argmax(predictions[0, masked_index])
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token)
assert predicted_token == 'henson'
