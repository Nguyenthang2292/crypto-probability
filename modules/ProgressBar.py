"""
Progress bar utility for displaying task progress.
"""
import threading
from colorama import Fore
from .utils import color_text


class ProgressBar:
    """
    Thread-safe progress bar for displaying task progress.
    
    Args:
        total: Total number of steps to complete
        label: Label to display before the progress bar
        width: Width of the progress bar in characters
    """
    def __init__(self, total: int, label: str = "Progress", width: int = 30):
        self.total = max(total, 1)
        self.label = label
        self.width = width
        self.current = 0
        self._lock = threading.Lock()

    def update(self, step: int = 1):
        """
        Update the progress bar by the specified number of steps.
        
        Args:
            step: Number of steps to advance (default: 1)
        """
        with self._lock:
            self.current = min(self.total, self.current + step)
            ratio = self.current / self.total
            filled = int(self.width * ratio)
            bar = "█" * filled + "-" * (self.width - filled)
            percent = ratio * 100
            print(f"\r{color_text(f'{self.label}: [{bar}] {self.current}/{self.total} ({percent:5.1f}%)', Fore.CYAN)}",
                  end='',
                  flush=True)

    def finish(self):
        """Complete the progress bar and print a newline."""
        self.update(0)
        print()

