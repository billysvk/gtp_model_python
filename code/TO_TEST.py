import os
import random
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup

# better train ;)
def train_model():
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Load text file as dataset
    with open('shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # Initialize GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize text and convert to torch tensors
    input_ids = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True).to(device)

    # Set training parameters
    train_batch_size = 4
    num_train_epochs = 500  # Increase number of epochs
    learning_rate = 3e-5   # Adjust learning rate

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(input_ids) * num_train_epochs // train_batch_size
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Train the model
    model.train()
    for epoch in range(num_train_epochs):
        epoch_loss = 0.0
        for i in range(0, len(input_ids)-1, train_batch_size):
            # Slice the input ids tensor to get the current batch
            batch_input_ids = input_ids[i:i+train_batch_size]
            # Create shifted labels for each input in the batch
            batch_labels = batch_input_ids.clone()
            batch_labels[:, :-1] = batch_labels[:, 1:]
            # Set label ids to -100 for padded tokens
            batch_labels[batch_labels == tokenizer.pad_token_id] = -100
            # Clear gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(input_ids=batch_input_ids, labels=batch_labels)
            loss = outputs[0]
            # Backward pass
            loss.backward()
            epoch_loss += loss.item()
            # Clip gradients to prevent exploding gradients problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters
            optimizer.step()
            scheduler.step()
        print('Epoch: {}, Loss: {:.4f}'.format(epoch+1, epoch_loss/len(input_ids)))

    # Save the trained model
    output_dir = './models1/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# def train_model():
#     # Set random seed for reproducibility
#     random.seed(42)
#     torch.manual_seed(42)

#     # Load text file as dataset
#     with open('shakespeare.txt', 'r', encoding='utf-8') as f:
#         text = f.read()

#     # Initialize GPT-2 tokenizer and model
#     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#     model = GPT2LMHeadModel.from_pretrained('gpt2')

#     # Set device to GPU if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Tokenize text and convert to torch tensors
#     input_ids = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True).to(device)

#     # Set training parameters
#     train_batch_size = 4
#     num_train_epochs = 3
#     learning_rate = 5e-5

#     # Initialize optimizer and scheduler
#     optimizer = AdamW(model.parameters(), lr=learning_rate)
#     total_steps = len(input_ids) * num_train_epochs // train_batch_size
#     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

#     # Train the model
#     model.train()
#     for epoch in range(num_train_epochs):
#         epoch_loss = 0.0
#         for i in range(0, len(input_ids)-1, train_batch_size):
#             # Slice the input ids tensor to get the current batch
#             batch_input_ids = input_ids[i:i+train_batch_size]
#             # Create shifted labels for each input in the batch
#             batch_labels = batch_input_ids.clone()
#             batch_labels[:, :-1] = batch_labels[:, 1:]
#             # Set label ids to -100 for padded tokens
#             batch_labels[batch_labels == tokenizer.pad_token_id] = -100
#             # Clear gradients
#             optimizer.zero_grad()
#             # Forward pass
#             outputs = model(input_ids=batch_input_ids, labels=batch_labels)
#             loss = outputs[0]
#             # Backward pass
#             loss.backward()
#             epoch_loss += loss.item()
#             # Clip gradients to prevent exploding gradients problem
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             # Update parameters
#             optimizer.step()
#             scheduler.step()
#         print('Epoch: {}, Loss: {:.4f}'.format(epoch+1, epoch_loss/len(input_ids)))

#     # Save the trained model
#     output_dir = './results/'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     model.save_pretrained(output_dir)
#     tokenizer.save_pretrained(output_dir)

def generate_text(prompt):
    # Load the trained tokenizer and model
    model_dir = './models1'
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    
    # Set the device to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # # Generate text based on the prompt
    # output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)

    # # Decode the generated text
    # generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

     # assert num_beams % num_beam_groups == 0, "`num_beams` should be divisible by `num_beam_groups` for group beam search."

    # output_ids = model.generate(input_ids, 
    #                              max_length=200,  # Adjust maximum length if needed
    #                              num_return_sequences=1,  # Generate multiple sequences
    #                              temperature=0.7,  # Adjust temperature if needed
    #                              top_k=50,  # Adjust top-k sampling if needed
    #                              top_p=0.9,  # Adjust top-p sampling (nucleus sampling) if needed
    #                              num_beams=1,  # Number of beams for beam search
    #                              num_beam_groups=1,  # Number of beam groups for diverse beam search
    #                              diversity_penalty=1,  # Diversity penalty for diverse beam search
    #                              )
    

    temperature = 0.5  # Adjust as needed (higher values for more diversity)
    top_k = 50        # Adjust as needed (lower values for more diversity)
    top_p = 0.9       # Adjust as needed (lower values for more diversity)

# Generate text based on the prompt
    output_ids = model.generate(
        input_ids,
        max_length=100,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=1
    )

      # # Decode the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

def main():
    # Train the model (only run this once)
    if not os.path.exists('./models1/'):
        train_model()

    # Generate text based on user prompt
    while True:
        prompt = input("Enter a prompt (type 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
        generated_text = generate_text(prompt)
        print("Generated Text:")
        print(generated_text)

if __name__ == "__main__":
    main()


# when Shakespeare was born?