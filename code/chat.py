import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def preprocess_text(text):
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]

    # Part-of-speech tagging
    pos_tags = pos_tag(tokens)

    return pos_tags

def respond_to_question(question):
    # Placeholder logic for responding to questions
    if 'who' in question and 'president' in question:
        return "The current president is Joe Biden."

    return "I'm sorry, I don't have information about that."

def main():
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Bot: Goodbye!")
            break

        # Preprocess user input
        processed_input = preprocess_text(user_input)

        # Placeholder logic for determining the type of question
        response = respond_to_question(processed_input)

        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
