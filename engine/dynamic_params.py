import json
import os

PARAMS_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'dynamic_parameters.json')

class DynamicParams:
    @staticmethod
    def load():
        """Load all dynamic parameters from config file."""
        with open(PARAMS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def save(params):
        """Save all dynamic parameters to config file."""
        with open(PARAMS_FILE, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)

    @staticmethod
    def add_param(section, param, label_en, label_he=None, description=None):
        """Add a new parameter to a section (payslip, attendance, contract)."""
        params = DynamicParams.load()
        if section not in params:
            params[section] = []
        # Prevent duplicates
        for p in params[section]:
            if p['param'] == param:
                p['label_en'] = label_en
                if label_he is not None:
                    p['label_he'] = label_he
                if description is not None:
                    p['description'] = description
                DynamicParams.save(params)
                return
        new_param = {'param': param, 'label_en': label_en}
        if label_he is not None:
            new_param['label_he'] = label_he
        if description is not None:
            new_param['description'] = description
        params[section].append(new_param)
        DynamicParams.save(params)

    @staticmethod
    def remove_param(section, param):
        """Remove a parameter from a section."""
        params = DynamicParams.load()
        if section in params:
            params[section] = [p for p in params[section] if p['param'] != param]
            DynamicParams.save(params)

    @staticmethod
    def get_params(section):
        """Get all parameters for a section."""
        params = DynamicParams.load()
        return params.get(section, [])

    @staticmethod
    def get_all_params():
        """Get all parameters for all sections as a flat dict: {section: [param, ...]}"""
        params = DynamicParams.load()
        return {k: [p['param'] for p in v] for k, v in params.items()}

    @staticmethod
    def get_all_param_labels():
        """Get all parameters with labels for all sections: {section: [{param, label_en, label_he, description}, ...]}"""
        return DynamicParams.load()
