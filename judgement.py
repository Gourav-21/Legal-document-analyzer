import json
from typing import List, Dict, Optional

class JudgementStorage:
    def __init__(self, file_path="judgements.json"):
        self.file_path = file_path
        self.judgements: Dict[str, Dict] = self._load_judgements()

    def _load_judgements(self) -> Dict[str, Dict]:
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            # Handle cases where the file is empty or contains invalid JSON
            return {}

    def _save_judgements(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.judgements, f, indent=4)

    def add_judgement(self, id: str, full_text: str) -> Dict:
        self.judgements[id] = {"id": id, "full_text": full_text}
        self._save_judgements()
        return self.judgements[id]

    def get_all_judgements(self) -> List[Dict]:
        return list(self.judgements.values())

    def get_judgement_by_id(self, judgement_id: str) -> Optional[Dict]:
        return self.judgements.get(judgement_id)

    def update_judgement(self, judgement_id: str, full_text: str) -> Optional[Dict]:
        if judgement_id in self.judgements:
            self.judgements[judgement_id]["full_text"] = full_text
            self._save_judgements()
            return self.judgements[judgement_id]
        return None

    def delete_judgement(self, judgement_id: str) -> bool:
        if judgement_id in self.judgements:
            del self.judgements[judgement_id]
            self._save_judgements()
            return True
        return False
    
    def format_judgements_for_prompt(self) -> str:
        formatted_judgements = [f"{i+1}. {judgement['full_text']}" for i, judgement in enumerate(self.judgements.values())]
        return "\n\n".join(formatted_judgements)