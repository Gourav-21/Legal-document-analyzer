import json
import os
from typing import List, Dict, Optional
from uuid import uuid4

class LetterFormatStorage:
    def __init__(self):
        self.file_path = "letter_format.json"
        self._ensure_storage_exists()
    
    def _ensure_storage_exists(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                json.dump({}, f)
    
    def _read_format(self) -> Dict:
        with open(self.file_path, 'r') as f:
            return json.load(f)
    
    def _write_format(self, format_data: Dict):
        with open(self.file_path, 'w') as f:
            json.dump(format_data, f, indent=2)
    
    def add_format(self, content: str) -> Dict:
        format_data = {"content": content}
        self._write_format(format_data)
        return format_data
    
    def get_format(self) -> Dict:
        return self._read_format()
    
    def update_format(self, content: str) -> Dict:
        format_data = {"content": content}
        self._write_format(format_data)
        return format_data
    
    def delete_format(self) -> bool:
        try:
            self._write_format({})
            return True
        except:
            return False