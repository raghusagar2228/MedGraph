from typing import List, Dict, Any
from langchain_core.tools import tool
from app.config import settings
from app.services.api_client import api_client
from app.services.neo4j_service import neo4j_service
from app.services.mock_data import MockDataRepository
from app.schemas.evidence import GraphRecord


@tool("search_patient_graph")
async def search_patient_graph(patient_id: str, query: str) -> List[Dict[str, Any]]:
    """Retrieve structured patient information from the Neo4j knowledge graph.

    Supports three explicit retrieval modes:
    - 'hybrid_live': Queries native Neo4j Bolt instance with Cypher.
    - 'external_api': Calls companion Express backend over HTTP.
    - 'mock': In-memory synthetic clinical knowledge store.
    """
    mode = settings.get_effective_retrieval_mode()

    if mode == "hybrid_live":
        records = await neo4j_service.search_patient_graph(patient_id=patient_id, query=query)
    elif mode == "external_api":
        records = await api_client.search_graph(patient_id=patient_id, query=query)
    else:
        records = MockDataRepository.search_graph(patient_id=patient_id, query=query)

    return [r.model_dump() for r in records]
