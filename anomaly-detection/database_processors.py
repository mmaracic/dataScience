#pylint: disable=W1514,W0603,W1203,C0114,C0116,C0115
import re
from abc import ABC, abstractmethod
import zipfile

class SampleSource(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def __str__(self):
        pass

    def get_path(self) -> str:
        return self.file_path

    @abstractmethod
    def split_log(self, log: str) -> list:
        pass

    @abstractmethod
    def get_log_idenfifier(self, log: str) -> str:
        pass

    def get_matching_part(self, s: str, pattern: str) -> str:
        match = re.search(pattern, s)
        return match.group(0) if match else None

    def read_log(self):
        if zipfile.is_zipfile(self.get_path()):
            return self.read_log_from_zip()
        else:
            return self.read_log_from_text()

    def read_log_from_zip(self):
        with zipfile.ZipFile(self.get_path()) as z:
            with z.open(...) as f:
                for line in f:
                    yield line.decode('utf-8')

    def read_log_from_text(self):
        with open(self.get_path(), 'r') as f:
            for line in f:
                yield line

    def read_range_of_logs(self, start: int, end: int):
        if zipfile.is_zipfile(self.get_path()):
            return self.read_range_of_logs_from_zip(start, end)
        else:
            return self.read_range_of_logs_from_text(start, end)

    def read_range_of_logs_from_zip(self, start: int, end: int):
        with zipfile.ZipFile(self.get_path()) as z:
            with z.open(...) as f:
                for _ in range(end):
                    line = f.readline().decode('utf-8')
                    if _ < start:
                        continue
                    else:
                        yield line

    def read_range_of_logs_from_text(self, start: int, end: int):
        with open(self.get_path(), 'r') as f:
            for _ in range(end):
                line = f.readline()
                if _ < start:
                    continue
                else:
                    yield line

class WindowsLogSource(SampleSource):
    date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
    time_pattern = re.compile(r"\d{2}:\d{2}:\d{2},")
    date_time_pattern = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

    def split_log(self, log: str) -> list:
        tokens = re.split(r'[\s~_\-/\\]+', log)
        return [
            w for w in tokens
            if not (self.date_pattern.fullmatch(w) or self.time_pattern.fullmatch(w))
        ]

    def get_log_idenfifier(self, log: str) -> str:
        return self.get_matching_part(log, self.date_time_pattern)
    
    def __str__(self):
        return "WindowsLogSource"

class WebServerLogSource(SampleSource):
    pattern = re.compile(r"\[(\d{2})/([A-Za-z]{3})/(\d{4}):(\d{2}):(\d{2}):(\d{2}) ([+-]\d{4})\]")

    def split_log(self, log: str) -> list:
        tokens = re.split(r'[\s~_\-/\\]+', log)
        return [
            w for w in tokens
            if not (self.pattern.fullmatch(w))
        ]
    def get_log_idenfifier(self, log: str) -> str:
        return self.get_matching_part(log, self.pattern)

    def __str__(self):
        return "WebServerLogSource"
