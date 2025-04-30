import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class BotReloader(FileSystemEventHandler):
    def __init__(self):
        self.process = None
        self.restart_delay = 1  # Delay in seconds before restarting
        self.last_restart = 0
        self.start_bot()

    def start_bot(self):
        """Start the bot process"""
        if self.process:
            self.stop_bot()
        
        logging.info("Starting bot...")
        self.process = subprocess.Popen([sys.executable, "bot/main.py"])
        self.last_restart = time.time()

    def stop_bot(self):
        """Stop the bot process"""
        if self.process:
            logging.info("Stopping bot...")
            self.process.terminate()
            self.process.wait()
            self.process = None

    def restart_bot(self):
        """Restart the bot with a delay to prevent rapid restarts"""
        current_time = time.time()
        if current_time - self.last_restart < self.restart_delay:
            return
        
        logging.info("Restarting bot...")
        self.start_bot()

    def on_modified(self, event):
        if event.is_directory:
            return

        # Only restart on Python file changes
        if Path(event.src_path).suffix == '.py':
            logging.info(f"Detected change in {event.src_path}")
            self.restart_bot()

    def __del__(self):
        self.stop_bot()

def main():
    # Create an observer and reloader
    observer = Observer()
    reloader = BotReloader()

    # Watch both the bot directory and its subdirectories
    observer.schedule(reloader, path='bot', recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Stopping development server...")
        observer.stop()
        reloader.stop_bot()
    
    observer.join()

if __name__ == "__main__":
    main() 