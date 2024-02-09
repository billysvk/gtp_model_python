from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('./results/')
model = GPT2LMHeadModel.from_pretrained('./results/')

# Define the prompt
prompt = "Once upon a time,"

# Tokenize the prompt
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Generate text based on the prompt
output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)

# Decode the generated text
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Print the generated text
print("Generated Text:")
print(generated_text)
