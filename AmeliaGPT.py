import subprocess
import sys

def install(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call(["pip", "install", package])

# Installing language model.
install("tensorflow")
install("transformers")

# Importing the downloaded modules.
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
    
    def generate_response(input_text, model, tokenizer):
        # Tokenize the input text
        input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

        # Generate a response from the model
        response = model.generate(input_ids=input_ids, max_length=1000, do_sample=True)

        # Decode the response and return it
        return tokenizer.decode(response[0], skip_special_tokens=True)
    
    while (True):
        
        # Get user input
        user_input = input("You: ")

        # Generate a response from the ChatGPT model
        response = generate_response(user_input, model, tokenizer)

        # Print the response
        print("AmeliaGPT: ", response)
        
        if "exit" in user_input.lower():
            sys.exit()
        
if __name__ == "__main__":
    main()

