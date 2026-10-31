from typing import List, Dict, Any
from langchain_core.tools import tool
from app.config import settings
from app.services.api_client import api_client
from app.services.neo4j_service import neo4j_service
from app.services.mock_data import MockDataRepository
from app.schemas.evidence import AdmissionRecord


@tool("get_admission_history")
async def get_admission_history(patient_id: str) -> List[Dict[str, Any]]:
    """Retrieve hospital admission and encounter history for a patient.

    Supports three explicit retrieval modes:
    - 'hybrid_live': Queries native Neo4j for Admission nodes.
    - 'external_api': Calls companion Express backend over HTTP.
    - 'mock': In-memory synthetic admission episodes.
    """
    mode = settings.get_effective_retrieval_mode()

    if mode == "hybrid_live":
        admissions = await neo4j_service.get_patient_admissions(patient_id=patient_id)
    elif mode == "external_api":
        admissions = await api_client.get_admissions(patient_id=patient_id)
    else:
        admissions = MockDataRepository.get_admissions(patient_id=patient_id)

    return [a.model_dump() for a in admissions]
