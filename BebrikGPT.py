# Importing system modules
import subprocess
import sys
import os
import threading
import time

# Creating a function to install packages that are not system-essential
def install(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call(["pip", "install", package])

# Installing modules for the NPL model
install("tensorflow")
install("transformers")
install("torch")
install("PyQt5")
install("pyperclip")

# Importing the modules needed
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt, QRunnable, QThreadPool
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QPlainTextEdit, QPushButton, QStyleFactory
import pyperclip
from transformers import GPT2Tokenizer, GPTNeoForCausalLM


# Class for multi-threading
class GetResponse(QRunnable):
    def __init__(self, input_text, tokenizer, model, text_edit):
        super().__init__()
        self.input_text = input_text
        self.tokenizer = tokenizer
        self.model = model
        self.text_edit = text_edit

    def run(self):
        # Generating a response using the GPT-3 model. max_length variable is set for the model response.
        # The longer it is the more time it may take for the model to generate a response.
        input_ids = self.tokenizer.encode(self.input_text, return_tensors='pt')
        bot_output = self.model.generate(input_ids, max_length=50, pad_token_id=self.tokenizer.eos_token_id)
        bot_response = self.tokenizer.decode(bot_output[0], skip_special_tokens=True)

        # Display the response in the chat window
        self.text_edit.insertPlainText('You: ' + self.input_text + '\n')
        self.text_edit.insertPlainText('BebrikGPT: ' + bot_response.replace(self.input_text, "") + '\n')


# Class for the GUI
class BebrikGPT(QMainWindow):
    def __init__(self):
        super(BebrikGPT, self).__init__()
        self.setWindowTitle("BebrikGPT")
        self.setFixedSize(900, 450)  # Set fixed window size
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
        self.setWindowIcon(QtGui.QIcon(os.path.abspath(icon_path)))
        self.initUI()

        # Create a thread pool
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(1)

        # Load the GPT-Neo 125M model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        self.model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")


    def initUI(self):
        self.setStyleSheet("background-color: #1E1E1E;")  # Set the GUI background

        self.text_edit = QPlainTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setGeometry(50, 50, 800, 250)
        self.text_edit.setStyleSheet("background-color: #1E1E1E; color: white;")

        self.input_edit = QPlainTextEdit(self)
        self.input_edit.setGeometry(50, 320, 550, 50)
        self.input_edit.setFocusPolicy(Qt.StrongFocus)
        self.input_edit.setStyleSheet("background-color: #1E1E1E; color: white;")
        self.input_edit.textChanged.connect(self.check_enter_pressed)

        # Create a Send button
        self.send_button = QPushButton('Send', self)
        self.send_button.setGeometry(620, 320, 80, 50)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                border: none;
                font-weight: bold;
                font-size: 14px;
                border-radius: 5px;
            }

            QPushButton:hover {
                background-color: #C0392B;
            }

            QPushButton:pressed {
                background-color: #E74C3C;
            }
        """)
        self.send_button.clicked.connect(self.send_input)

        # Create a Clear button
        self.clear_button = QPushButton('Clear', self)
        self.clear_button.setGeometry(710, 320, 80, 50)
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                border: none;
                font-weight: bold;
                font-size: 14px;
                border-radius: 5px;
            }

            QPushButton:hover {
                background-color: #C0392B;
            }

            QPushButton:pressed {
                background-color: #E74C3C;
            }
        """)
        self.clear_button.clicked.connect(self.clear_input)

        # Create a Copy button
        self.copy_button = QPushButton('Copy', self)
        self.copy_button.setGeometry(800, 320, 80, 50)
        self.copy_button.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                border: none;
                font-weight: bold;
                font-size: 14px;
                border-radius: 5px;
            }

            QPushButton:hover {
                background-color: #C0392B;
            }

            QPushButton:pressed {
                background-color: #E74C3C;
            }
        """)
        self.copy_button.clicked.connect(self.copy_output)

        self.setStyleSheet("background-color: #1E1E1E;")  # Set the GUI background

        self.show()


    def send_input(self):
        input_text = self.input_edit.toPlainText()
        if input_text:
            self.input_edit.clear()
            runnable = GetResponse(input_text, self.tokenizer, self.model, self.text_edit)
            self.threadpool.start(runnable)


    def check_enter_pressed(self):
        if "\n" in self.input_edit.toPlainText():
            self.send_input()

    def copy_output(self):
        output_text = self.text_edit.toPlainText()
        if output_text:
            pyperclip.copy(output_text)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.send_input()

    def copy_output(self):
        output_text = self.text_edit.toPlainText()
        if output_text:
            pyperclip.copy(output_text)

    def clear_input(self):
        self.text_edit.clear()
        self.input_edit.clear()

# Initalizing
if __name__ == '__main__':
    app = QApplication(sys.argv)
    bebrikGPT = BebrikGPT()
    bebrikGPT.show()
    sys.exit(app.exec_())