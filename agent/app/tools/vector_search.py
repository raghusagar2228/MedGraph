from typing import List, Dict, Any
from langchain_core.tools import tool
from app.config import settings
from app.services.api_client import api_client
from app.services.vector_service import vector_service
from app.services.mock_data import MockDataRepository
from app.schemas.evidence import ClinicalNote


@tool("search_clinical_notes")
async def search_clinical_notes(patient_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve unstructured clinical notes using vector semantic search.

    Supports three explicit retrieval modes:
    - 'hybrid_live': Uses native FastEmbed ONNX dense vectors and cosine similarity.
    - 'external_api': Calls companion Express vector endpoint over HTTP.
    - 'mock': In-memory synthetic notes with keyword scoring heuristic.
    """
    mode = settings.get_effective_retrieval_mode()

    if mode == "hybrid_live":
        notes = vector_service.search(patient_id=patient_id, query=query, top_k=top_k)
    elif mode == "external_api":
        notes = await api_client.search_vector(patient_id=patient_id, query=query, top_k=top_k)
    else:
        notes = MockDataRepository.search_notes(patient_id=patient_id, query=query, top_k=top_k)

    return [n.model_dump() for n in notes]
