import gpt_2_simple as gpt2
from PyPDF2 import PdfReader
import os
import tensorflow as tf
import requests

# Load knowledge from PDF file
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=10)
# tf.config.optimizer.set_experimental_options({"mlir": True})
# pdf_path = input("Enter the path to the PDF file: ")
# knowledge_text = extract_text_from_pdf(pdf_path)
# knowledge_text = "My name is Vasileios"

model_name = "124M"  # You can also try "355M" for a larger model
# model_folder = os.path.join("models", model_name)
model_folder = os.path.join("checkpoint", model_name)
# Fine-tune GPT-2 on the knowledge text
# Check if the GPT-2 model checkpoint for the specified model_name exists
# if not os.path.exists(model_folder):
#     # If not, download the GPT-2 model
#     gpt2.download_gpt2(model_name=model_name)
# else:
#     print(f"GPT-2 model {model_name} is already downloaded.")
# sess = gpt2.start_tf_sess()  # Start a new TensorFlow session
# gpt2.reset_session(sess)  # Reset the TensorFlow session
# sess = gpt2.start_tf_sess()  # Use GPU if available
# Example adjustment to avoid division by zero
# gpt2.finetune(sess, knowledge_text, model_name=model_name, steps=1, batch_size=1, print_every=1)  # Adjust 'steps' and 'batch_size' values as needed
# print("skata")
# Save the fine-tuned model
if not os.path.isdir(os.path.join("models", model_name)):
	print(f"Downloading {model_name} model...")
	# gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/


file_name = "shakespeare.txt"
if not os.path.isfile(file_name):
	url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
	data = requests.get(url)

	with open(file_name, 'w') as f:
		f.write(data.text)
          
sess = gpt2.start_tf_sess()
# gpt2.finetune(sess,
#               file_name,
#               model_name=model_name,
#               steps=3)   # steps is max number of training steps

# gpt2.generate(sess)          
# gpt2.save_gpt2(model_name=model_name)
# Generate text using the fine-tuned model
# generated_text = gpt2.generate(sess)
# Print or use the generated text as needed
# print(generated_text)
# Close the TensorFlow session when done
# gpt2.reset_session(sess)
# single_text = gpt2.generate(sess, return_as_list=True)[0]
# print(single_text)

def generate_response(prompt):
    # Generate response
    response = gpt2.generate(sess, prompt, model_name=model_name, return_as_list=True)[0]
    return response

def main():
    print("ChatBot: Hello! I'm your local GPT-like chatbot. Type 'exit' to end the conversation.")
    
    # gpt2.reset_session()  # Reset the TensorFlow session
    # sess = gpt2.start_tf_sess()  # Start a new TensorFlow session

    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("ChatBot: Goodbye!")
                break

            response = generate_response(user_input)
            print(f"ChatBot: {response}")

    finally:
        gpt2.reset_session(sess)  # Reset the TensorFlow session when done

if __name__ == "__main__":
    main()


#pip install PyPDF2
#pip install tensorflow==2.15.0
#https://openai.com/research/gpt-2-1-5b-release