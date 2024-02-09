import sys
import re
from pdfminer.high_level import extract_text
import logging

logging.basicConfig(filename='chatbot.log', level=logging.DEBUG)
logging.debug('Debug message')


class ChatBot:
    def __init__(self):
        self.knowledge_base = {}

    def learn_from_pdf(self, pdf_path):
        print(f"Learning from PDF: {pdf_path}")
        # ... existing code ...
        text = extract_text(pdf_path)
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            # Assume the first word in the sentence is a keyword/topic
            words = sentence.strip().split()
            if words:
                topic = words[0].lower()
                if topic not in self.knowledge_base:
                    self.knowledge_base[topic] = []
                self.knowledge_base[topic].append(sentence)

    def generate_response(self, user_input):
        # Assume the first word in the user input is a keyword/topic
        print(f"Generating response for: {user_input}")
        # ... existing code ...
        words = user_input.strip().split()
        if words:
            topic = words[0].lower()
            if topic in self.knowledge_base:
                return self.knowledge_base[topic]
        return ["I'm sorry, I don't have information about that."]
    
    def print_knowledge_base(self):
        for topic, sentences in self.knowledge_base.items():
            print(f"Topic: {topic}")
            for sentence in sentences:
                print(f"  - {sentence}")
            print()

def main():
    # if len(sys.argv) != 2:
    #     print("Usage: python pdf_chatbot.py <pdf_path>")
    #     return
    # pdf_path = sys.argv[1]
    chatbot = ChatBot()
    pdf_path = input("Enter the path to the PDF file: ")
    chatbot.learn_from_pdf(pdf_path)

    print("ChatBot: Hello! I'm your PDF-learning chatbot. Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("ChatBot: Goodbye!")
            break
        elif user_input.lower() == 'print knowledge base':
            chatbot.print_knowledge_base()
            continue

        responses = chatbot.generate_response(user_input)
        for response in responses:
            print(f"ChatBot: {response}")

if __name__ == "__main__":
    main()

#create an executable
#pip install pyinstaller
#pyinstaller --onefile pdf_chatbot.py

#run from cmd
# pdf_learn.exe C:\Users\billysvk\Desktop\test1.pdf