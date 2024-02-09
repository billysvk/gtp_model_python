from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from PyPDF2 import PdfReader

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load knowledge from PDF file
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

pdf_path = input("Enter the path to the PDF file: ")
knowledge_text = extract_text_from_pdf(pdf_path)

# Tokenize and encode the knowledge text
knowledge_tokens = tokenizer.encode(knowledge_text, return_tensors="pt")

def generate_response(prompt, knowledge_tokens):
    # Tokenize and encode the prompt
    input_tokens = tokenizer.encode(prompt, return_tensors="pt")

    # Concatenate knowledge_tokens and input_tokens
    input_tokens = torch.cat([knowledge_tokens, input_tokens], dim=1)

    # Generate response
    output = model.generate(input_tokens, max_length=10000, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, do_sample=True)
    #outputs = model.generate(input_ids, max_length=100, do_sample=True, top_p=0.95)
    # Decode the generated output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def main():
    print("ChatBot: Hello! I'm your local GPT-like chatbot. Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("ChatBot: Goodbye!")
            break

        response = generate_response(user_input, knowledge_tokens)
        print(f"ChatBot: {response}")

if __name__ == "__main__":
    main()
