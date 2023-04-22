import subprocess

def install(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call(["pip", "install", package])

# Installing language model.
install("tensorflow")
install("transformers")
install("torch")
install("PyQt5")
install("pyperclip")

# Importing the downloaded modules.
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QPlainTextEdit, QPushButton
import threading
import time
import pyperclip
from transformers import AutoModelForCausalLM, AutoTokenizer

class BebrikGPT(QMainWindow):
    def __init__(self):
        super(BebrikGPT, self).__init__()
        self.setWindowTitle("BebrikGPT")
        self.setGeometry(200, 200, 800, 500)
        self.initUI()
        self.threadpool = []

        # Load the DialoGPT-medium model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    def initUI(self):
        self.text_edit = QPlainTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setGeometry(50, 50, 700, 300)

        self.input_edit = QPlainTextEdit(self)
        self.input_edit.setGeometry(50, 370, 500, 50)

        self.send_button = QPushButton('Send', self)
        self.send_button.setGeometry(570, 370, 100, 50)
        self.send_button.clicked.connect(self.send_input)

        self.copy_button = QPushButton('Copy', self)
        self.copy_button.setGeometry(690, 370, 100, 50)
        self.copy_button.clicked.connect(self.copy_output)

    def send_input(self):
        input_text = self.input_edit.toPlainText()
        if input_text:
            self.input_edit.clear()
            t1 = threading.Thread(target=self.get_response, args=(input_text,))
            self.threadpool.append(t1)
            t1.start()

    def get_response(self, input_text):
        # Generate a response using the DialoGPT-medium model
        input_ids = self.tokenizer.encode(input_text + self.tokenizer.eos_token, return_tensors='pt')
        bot_output = self.model.generate(input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        bot_response = self.tokenizer.decode(bot_output[0], skip_special_tokens=True)

        # Display the response in the chat window
        self.text_edit.insertPlainText('You: ' + input_text + '\n')
        self.text_edit.insertPlainText('BebrikGPT: ' + bot_response + '\n')

    def copy_output(self):
        output_text = self.text_edit.toPlainText()
        if output_text:
            pyperclip.copy(output_text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    chatbot = BebrikGPT()
    chatbot.show()
    sys.exit(app.exec_())
