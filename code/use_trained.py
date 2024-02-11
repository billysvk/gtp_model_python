

#################################################################################
#example code for a text generation app using the Hugging Face transformers library and the GPT-2 model:
# from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM



# Example usage
tokenizer = AutoTokenizer.from_pretrained("openai-community/openai-gpt")
model = TFAutoModelForCausalLM.from_pretrained("openai-community/openai-gpt")
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

