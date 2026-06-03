from PySide6.QtGui import QGuiApplication

# Ensure an application instance exists first
app = QGuiApplication.instance()

screens = QGuiApplication.screens()
for index, screen in enumerate(screens):
    print(f"Screen {index}: {screen.name()} - {screen.size().width()}x{screen.size().height()}")
