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

# Importing the downloaded modules.

from PyQt5 import QtWidgets, QtGui, QtCore
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class AmeliaGPT(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Load pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/dialogpt-large")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/dialogpt-large")

        # Create user input and output text boxes
        self.user_input = QtWidgets.QTextEdit()
        self.user_input.setFixedHeight(100)
        self.user_input.setFont(QtGui.QFont("Arial", 12))
        self.chat_output = QtWidgets.QTextEdit()
        self.chat_output.setReadOnly(True)
        self.chat_output.setFont(QtGui.QFont("Courier New", 10))
        self.chat_output.setStyleSheet("QTextEdit {background-color: black; color: white;}")
        self.chat_output.textCursor().insertHtml("<font color='red'>AmeliaGPT:</font> Type your message and press 'Send'.")

        # Create button for generating response
        self.send_button = QtWidgets.QPushButton("Send")
        self.send_button.setFont(QtGui.QFont("Arial", 12))
        self.send_button.clicked.connect(self.generate_response)

        # Create button for copying output
        self.copy_button = QtWidgets.QPushButton("Copy")
        self.copy_button.setFont(QtGui.QFont("Arial", 12))
        self.copy_button.clicked.connect(self.copy_text)

        # Create layout for window
        self.layout = QtWidgets.QGridLayout()
        self.layout.addWidget(self.user_input, 0, 0, 1, 2)
        self.layout.addWidget(self.send_button, 1, 1)
        self.layout.addWidget(self.chat_output, 2, 0, 1, 2)
        self.layout.addWidget(self.copy_button, 3, 1)
        self.setLayout(self.layout)

        # Set window properties
        self.setGeometry(300, 300, 600, 500)
        self.setWindowTitle("AmeliaGPT")
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setStyleSheet("background-color: #252526; color: white;")

        # Set focus on user input box
        self.user_input.setFocus()

    def generate_response(self):
        # Get user input and generate response
        prompt = self.user_input.toPlainText()
        if prompt:
            response = self.chat_output.toPlainText()
            if response:
                response += "\n\n"
            response += f"<font color='green'>You: {prompt}</font>\n\n"
            response += f"<font color='red'>AmeliaGPT: {self._generate_response(prompt)}</font>\n\n"
            self.chat_output.setText(response)
            self.user_input.clear()

    def _generate_response(self, prompt):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(input_ids=input_ids, max_length=1000, do_sample=True, top_p=0.92, top_k=50)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response.replace('\n', '<br>')

    def copy_text(self):
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(self.chat_output.toPlainText())

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return and not event.modifiers():
            self.send_button.click()
        else:
            super().keyPressEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    app.setStyle("Fusion")
    ameliagpt = AmeliaGPT()
    ameliagpt.show()