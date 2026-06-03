import sys
import win32gui
import win32con
from PySide6.QtGui import QGuiApplication


def _get_screens():
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)
    return app.screens()


def _find_window_by_title(partial_title):
    """Return the first visible window whose title contains partial_title (case-insensitive)."""
    matches = []

    def callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            if partial_title.lower() in win32gui.GetWindowText(hwnd).lower():
                matches.append(hwnd)

    win32gui.EnumWindows(callback, None)
    return matches[0] if matches else None


def list_visible_windows():
    """Print the titles of all currently visible windows."""
    titles = []

    def callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            text = win32gui.GetWindowText(hwnd)
            if text:
                titles.append(text)

    win32gui.EnumWindows(callback, None)
    for i, title in enumerate(titles):
        print(f"{i}: {title}")
    return titles


def move_window_to_monitor(title, screen_number, maximize_window=True):
    """
    Move a visible window whose title contains `title` to the given screen.

    Parameters
    ----------
    title : str
        Full or partial window title (case-insensitive).
    screen_number : int
        0-based index of the target screen. Values beyond the last
        screen are clamped to the last available screen.
    maximize_window : bool
        If True, maximize the window after moving. Default is True.

    Returns
    -------
    bool
        True if the window was moved, False otherwise.
    """
    screens = _get_screens()
    if not screens:
        print("No screens detected.")
        return False

    screen_index = min(screen_number, len(screens) - 1)
    target_screen = screens[screen_index]

    hwnd = _find_window_by_title(title)
    if hwnd is None:
        print(f"No visible window found with title containing: '{title}'")
        return False

    geo = target_screen.geometry()

    # Restore minimized windows before moving
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

    if maximize_window:
        # Move to target screen first, then maximize so it fills that screen
        win32gui.SetWindowPos(
            hwnd, None,
            geo.x(), geo.y(), geo.width(), geo.height(),
            win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE,
        )
        win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
    else:
        # Preserve current window size
        rect = win32gui.GetWindowRect(hwnd)
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        win32gui.SetWindowPos(
            hwnd, None,
            geo.x(), geo.y(), width, height,
            win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE,
        )

    print(
        f"Moved '{win32gui.GetWindowText(hwnd)}' to screen {screen_index} "
        f"({target_screen.name()}) at ({geo.x()}, {geo.y()})"
        + (" [maximized]" if maximize_window else "")
    )
    return True


if __name__ == "__main__":
    # Usage: move_window_to_monitor("Notepad", 1)
    import sys as _sys
    if len(_sys.argv) == 3:
        move_window_to_monitor(_sys.argv[1], int(_sys.argv[2]))
    else:
        list_visible_windows()
        print("Usage: python MoveWindowToMonitor.py <window title> <screen number>")
