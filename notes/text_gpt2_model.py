from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.default_chat_template
model = GPT2Model.from_pretrained('gpt2')
text = "What do you think about having a new bike."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

# https://huggingface.co/docs/transformers/main/chat_templating