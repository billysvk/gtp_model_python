#use this code to convert PyTorch to TensorFlow model
from transformers import TFGPT2LMHeadModel, GPT2LMHeadModel, GPT2Tokenizer, TFGPT2LMHeadModel, PreTrainedTokenizerFast
import tensorflow as tf
# tf_model_dir = './results_tf/'

# # Load the PyTorch model
# pytorch_model_dir = './results/'
# model = GPT2LMHeadModel.from_pretrained(pytorch_model_dir)

# # Convert the PyTorch model to TensorFlow format
# tf_model = TFGPT2LMHeadModel.from_pretrained(pytorch_model_dir)

# # Save the TensorFlow model
# tf_model.save_pretrained(tf_model_dir)

#################################################################################
#################################################################################
#below, the method how to load the transformed model
# These files are all necessary for loading and using the converted TensorFlow model. You can use them as follows:
# from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
# Load the tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained(tf_model_dir)
# Load the TensorFlow model
# model = TFGPT2LMHeadModel.from_pretrained(tf_model_dir)


#################################################################################
# Here's an example of how you can train a new tokenizer using the GPT2Tokenizer class:

# Prepare training data (replace 'train.txt' with the path to your training data)
training_data_path = 'shakespeare.txt'

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Train tokenizer on the training data
# tokenizer.train(files=training_data_path)

# Save tokenizer files to a directory
output_dir = 'tokenizer_output/'
tokenizer.save_pretrained(output_dir)


#################################################################################
#example code for a text generation app using the Hugging Face transformers library and the GPT-2 model:
# from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
# import tensorflow as tf

# Load the tokenizer and TensorFlow model
model_dir = 'C:/Users/billysvk/Desktop/py_projects/code/results_tf/'  # Assuming this is your model directory
tokenizer = GPT2Tokenizer.from_pretrained('tokenizer_output/')  # Load the tokenizer from the trained directory
model = TFGPT2LMHeadModel.from_pretrained(model_dir)

# Set the device to GPU if available
device = "cuda" if tf.test.is_gpu_available() else "cpu"
model = model.to(device)

# Function to generate text
def generate_text(prompt, max_length=100, temperature=0.7):
    input_ids = tokenizer.encode(prompt, return_tensors='tf').to(device)
    output_ids = model.generate(input_ids, max_length=max_length, temperature=temperature)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# Interactive text generation loop
while True:
    prompt = input("Enter a prompt (or 'exit' to quit): ")
    if prompt.lower() == 'exit':
        break
    generated_text = generate_text(prompt)
    print("Generated Text:")
    print(generated_text)

