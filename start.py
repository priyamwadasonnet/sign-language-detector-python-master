import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import subprocess  # To run another Python script

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Set up the window
        self.setWindowTitle("PyQt5 Image and Button Example")
        self.setGeometry(100, 100, 400, 300)

        # Layout to organize the widgets
        layout = QVBoxLayout()

        # Add an image to the window
        self.image_label = QLabel(self)
        pixmap = QPixmap('path/to/your/image.jpg')  # Replace with your image file path
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)

        # Add a button to open another Python file
        self.button = QPushButton("Open Another Script", self)
        self.button.clicked.connect(self.open_other_script)

        # Add the widgets to the layout
        layout.addWidget(self.image_label)
        layout.addWidget(self.button)

        # Set the layout for the window
        self.setLayout(layout)

    def open_other_script(self):
        """Function to open another Python file when button is clicked."""
        # Close the current window before opening the other script
        self.close()

        # Open the other Python script
        subprocess.run([sys.executable, 'inference_classifier.py'])  # Replace 'other_script.py' with your file name


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Create and show the main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
