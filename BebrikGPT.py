import subprocess

import sys
import os
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QPlainTextEdit, QPushButton, QStyleFactory
import threading
import time
import pyperclip
import transformers

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

class BebrikGPT(QMainWindow):
    def __init__(self):
        super(BebrikGPT, self).__init__()
        self.setWindowTitle("BebrikGPT")
        self.setGeometry(200, 200, 800, 500)
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
        self.setWindowIcon(QtGui.QIcon(os.path.abspath(icon_path)))
        self.initUI()
        self.threadpool = []

        # Load the GPT-3 model and tokenizer
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        self.model = transformers.GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

    def initUI(self):
        self.text_edit = QPlainTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setGeometry(50, 50, 700, 300)
        self.text_edit.setStyleSheet("background-color: #1E1E1E; color: white;")

        self.input_edit = QPlainTextEdit(self)
        self.input_edit.setGeometry(50, 370, 500, 50)
        self.input_edit.setFocusPolicy(Qt.StrongFocus)
        self.input_edit.setStyleSheet("background-color: #1E1E1E; color: white;")
        self.input_edit.textChanged.connect(self.check_enter_pressed)

        self.send_button = QPushButton('Send', self)
        self.send_button.setGeometry(570, 370, 100, 50)
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

        self.copy_button = QPushButton('Copy', self)
        self.copy_button.setGeometry(690, 370, 100, 50)
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

    def send_input(self):
        input_text = self.input_edit.toPlainText()
        if input_text:
            self.input_edit.clear()
            self.get_response(input_text)

    def get_response(self, input_text):
        # Generate a response using the GPT-3 model
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        bot_output = self.model.generate(input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        bot_response = self.tokenizer.decode(bot_output[0], skip_special_tokens=True)

        # Display the response in the chat window
        self.text_edit.insertPlainText('You: ' + input_text + '\n')
        self.text_edit.insertPlainText('BebrikGPT: ' + bot_response.replace(input_text, "") + '\n')

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    chatbot = BebrikGPT()
    chatbot.show()
    sys.exit(app.exec_())
