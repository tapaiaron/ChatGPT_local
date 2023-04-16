import subprocess

def install(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call(["pip", "install", package])

# Installing language model.
install("transformers torch")

# Importing the downloaded modules.
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    while True:
        user_input = input("You: ")
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        chat_response = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        print("ChatGPT: " + tokenizer.decode(chat_response[:, input_ids.shape[-1]:][0], skip_special_tokens=True))

if __name__ == "__main__":
    main()

