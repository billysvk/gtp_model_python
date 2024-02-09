import os
import csv
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

#Pre-process the Data
def get_conversations():
    conversations = []
    with open(os.path.join('cornell_movie_dialogs_corpus',
              'movie_titles_metadata.txt'), 'r', encoding='iso-8859-1') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(' +++$+++ ')
            if len(parts) == 5:
                conv_id = parts[0]
                line_text = parts[4]
                conversations.append((conv_id, line_text))
    return conversations

# Train the Model
def train():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = TFGPT2LMHeadModel.from_pretrained('gpt2')

    # Load the conversation dataset
    conversations = get_conversations()

    # Tokenize the conversations and create input/output pairs
    input_ids = []
    output_ids = []
    for i in range(len(conversations) - 1):
        input_text = conversations[i][1]
        output_text = conversations[i + 1][1]
        input_tokenized = tokenizer.encode(input_text,
                add_special_tokens=False)
        output_tokenized = tokenizer.encode(output_text,
                add_special_tokens=False)
        input_ids.append(input_tokenized)
        output_ids.append(output_tokenized)

    # Pad the input/output pairs to the same length
    max_length = max(len(ids) for ids in input_ids + output_ids)
    input_ids = \
        tf.keras.preprocessing.sequence.pad_sequences(input_ids,
            maxlen=max_length, padding='post')
    output_ids = \
        tf.keras.preprocessing.sequence.pad_sequences(output_ids,
            maxlen=max_length, padding='post')

    # Define the training parameters
    batch_size = 16
    epochs = 10
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

    # Compile the model
    model.compile(optimizer=optimizer, loss=model.compute_loss)
# Print the shapes and types of input_ids and output_ids
    print("Shape of input_ids:", input_ids.shape)
    print("Type of input_ids:", input_ids.dtype)
    print("Shape of output_ids:", output_ids.shape)
    print("Type of output_ids:", output_ids.dtype)

    # Train the model
    model.fit(input_ids, output_ids, batch_size=batch_size,
              epochs=epochs)

    # Save the trained model
    model.save_pretrained('chatbot_model')
    return max_length

# Test the Model    
def test(max_length):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = TFGPT2LMHeadModel.from_pretrained("chatbot_model")

    while True:
        input_text = input("User: ")
        input_tokenized = tokenizer.encode(input_text, add_special_tokens=False)
        input_ids = tf.keras.preprocessing.sequence.pad_sequences([input_tokenized], maxlen=max_length, padding="post")
        output_ids = model.generate(input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("Bot:", output_text)    

def main():
    print("ChatBot: Hello! I'm your local GPT-like chatbot. Type 'exit' to end the conversation.")
    max_length = train()  # Store the returned max_length
    test(max_length)  # Pass max_length to test()

if __name__ == "__main__":
    main()     