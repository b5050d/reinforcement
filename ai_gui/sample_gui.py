import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout

class SimpleWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyQt5 Simple GUI")
        self.setGeometry(100, 100, 300, 150)

        self.label = QLabel("Hello, PyQt5!", self)
        self.button = QPushButton("Click Me", self)
        self.button.clicked.connect(self.on_button_click)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)

        self.setLayout(layout)

    def on_button_click(self):
        self.label.setText("Button clicked!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimpleWindow()
    window.show()
    sys.exit(app.exec_())
