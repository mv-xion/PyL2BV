"""
    Defining a logger class
"""
import atexit


class Logger:
    def __init__(self, path):
        self.path = path
        # Opening logfile for appending
        self.log_file_id = open(f"{self.path}_logfile.log", 'w')
        atexit.register(self.close)

    def open(self):
        if self.log_file_id.closed:
            self.log_file_id = open(f"{self.path}_logfile.log", 'a')

    def log_message(self, message):
        # Log information to logfile
        self.log_file_id.write(message)

    def close(self):
        if self.log_file_id:
            self.log_file_id.close()
