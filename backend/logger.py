from datetime import datetime
import collections

class EventLogger:
    def __init__(self, max_size=50):
        self.logs = collections.deque(maxlen=max_size)

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.logs.append(entry)
        print(entry, flush=True) # Ensure immediate write to log file

    def get_logs(self):
        return list(self.logs)

# Global instance
event_logger = EventLogger()
