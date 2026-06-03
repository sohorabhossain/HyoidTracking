import sys
from PySide6.QtGui import QGuiApplication

app = QGuiApplication.instance() or QGuiApplication(sys.argv)

for index, screen in enumerate(app.screens()):
    size = screen.size()
    print(f"Screen {index}: {screen.name()} - {size.width()}x{size.height()}")
