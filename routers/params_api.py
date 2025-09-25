from fastapi import APIRouter, HTTPException
from typing import List, Dict, Optional
from pydantic import BaseModel
import sys
import os

# Add engine directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'engine'))
from engine.dynamic_params import DynamicParams

router = APIRouter()

# Pydantic models for API validation
class ParameterCreate(BaseModel):
    param: str
    label_en: str
    label_he: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = "number"

class ParameterUpdate(BaseModel):
    param: Optional[str] = None
    label_en: Optional[str] = None
    label_he: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None

class ParameterResponse(BaseModel):
    param: str
    label_en: str
    label_he: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = "number"

# --- CRUD for dynamic parameters ---

@router.get('/dynamic-params', response_model=Dict[str, List[ParameterResponse]])
def get_all_dynamic_params():
    """Get all dynamic parameters organized by sections."""
    try:
        params = DynamicParams.load()
        return params
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading parameters: {str(e)}")

@router.get('/dynamic-params/{section}', response_model=List[ParameterResponse])
def get_params_by_section(section: str):
    """Get all parameters for a specific section (payslip, attendance, contract, employee)."""
    try:
        params = DynamicParams.get_params(section)
        return params
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading parameters for section {section}: {str(e)}")

@router.post('/dynamic-params/{section}', response_model=ParameterResponse)
def create_parameter(section: str, param_data: ParameterCreate):
    """Add a new parameter to a section."""
    try:
        # Trim input strings
        param = param_data.param.strip() if param_data.param else param_data.param
        label_en = param_data.label_en.strip() if param_data.label_en else param_data.label_en
        label_he = param_data.label_he.strip() if param_data.label_he else param_data.label_he
        description = param_data.description.strip() if param_data.description else param_data.description
        
        # Check if parameter already exists
        existing_params = DynamicParams.get_params(section)
        if any(p['param'] == param for p in existing_params):
            raise HTTPException(status_code=400, detail=f"Parameter '{param}' already exists in section '{section}'")

        DynamicParams.add_param(section, param, label_en, label_he, description, param_data.type)
        return {"param": param, "label_en": label_en, "label_he": label_he, "description": description, "type": param_data.type}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating parameter: {str(e)}")

@router.put('/dynamic-params/{section}/{param_name}', response_model=ParameterResponse)
def update_parameter(section: str, param_name: str, param_data: ParameterUpdate):
    """Update an existing parameter in a section."""
    try:
        params = DynamicParams.load()
        if section not in params:
            raise HTTPException(status_code=404, detail=f"Section '{section}' not found")

        # Find the parameter
        param_index = None
        for i, p in enumerate(params[section]):
            if p['param'] == param_name:
                param_index = i
                break

        if param_index is None:
            raise HTTPException(status_code=404, detail=f"Parameter '{param_name}' not found in section '{section}'")

        # Update the parameter with trimmed values
        if param_data.param is not None:
            params[section][param_index]['param'] = param_data.param.strip()
        if param_data.label_en is not None:
            params[section][param_index]['label_en'] = param_data.label_en.strip()
        if param_data.label_he is not None:
            params[section][param_index]['label_he'] = param_data.label_he.strip()
        if param_data.description is not None:
            params[section][param_index]['description'] = param_data.description.strip()
        if param_data.type is not None:
            params[section][param_index]['type'] = param_data.type

        DynamicParams.save(params)
        return params[section][param_index]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating parameter: {str(e)}")

@router.delete('/dynamic-params/{section}/{param_name}')
def delete_parameter(section: str, param_name: str):
    """Remove a parameter from a section."""
    try:
        # Check if parameter exists
        existing_params = DynamicParams.get_params(section)
        if not any(p['param'] == param_name for p in existing_params):
            raise HTTPException(status_code=404, detail=f"Parameter '{param_name}' not found in section '{section}'")

        DynamicParams.remove_param(section, param_name)
        return {"message": f"Parameter '{param_name}' removed from section '{section}'"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting parameter: {str(e)}")

# Additional utility endpoints
@router.get('/dynamic-params-sections')
def get_available_sections():
    """Get list of available parameter sections."""
    try:
        params = DynamicParams.load()
        return {"sections": list(params.keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading sections: {str(e)}")

@router.get('/dynamic-params-flat')
def get_all_params_flat():
    """Get all parameters as a flat list with section information."""
    try:
        params = DynamicParams.load()
        flat_params = []
        for section, param_list in params.items():
            for param in param_list:
                flat_params.append({
                    "section": section,
                    "param": param["param"],
                    "label_en": param["label_en"],
                    "label_he": param.get("label_he"),
                    "description": param.get("description")
                })
        return {"parameters": flat_params}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading flat parameters: {str(e)}")