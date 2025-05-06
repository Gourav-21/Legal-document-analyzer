from typing import List, Dict
from datetime import datetime
import json
import os
import uuid

class LaborLawStorage:
    def __init__(self, storage_file: str = "labor_laws.json"):
        self.storage_file = storage_file
        self.laws: List[Dict] = self._load_laws()
    
    def _load_laws(self) -> List[Dict]:
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _save_laws(self) -> None:
        with open(self.storage_file, 'w', encoding='utf-8') as f:
            json.dump(self.laws, f, indent=2, ensure_ascii=False)
    
    def add_law(self, law_text: str) -> Dict:
        law = {"id": str(uuid.uuid4()), "text": law_text}
        self.laws.append(law)
        self._save_laws()
        return law
    
    def get_all_laws(self) -> List[Dict]:
        return self.laws
    
    def delete_law(self, law_id: str) -> bool:
        for i, law in enumerate(self.laws):
            if law["id"] == law_id:
                self.laws.pop(i)
                self._save_laws()
                return True
        return False
    
    def update_law(self, law_id: str, new_text: str) -> Dict:
        for law in self.laws:
            if law["id"] == law_id:
                law["text"] = new_text
                self._save_laws()
                return law
        return None

    def format_laws_for_prompt(self) -> str:
        formatted_laws = [f"{i+1}. {law['text']}" for i, law in enumerate(self.laws)]
        return "\n\n".join(formatted_laws)