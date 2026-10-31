import asyncio
import logging
from typing import List, Tuple
from app.config import settings
from app.services.api_client import api_client
from app.services.neo4j_service import neo4j_service
from app.services.vector_service import vector_service
from app.services.mock_data import MockDataRepository
from app.schemas.evidence import GraphRecord, ClinicalNote, AdmissionRecord
from app.schemas.responses import InvestigationPlan

logger = logging.getLogger(__name__)


class RetrievalRouter:
    """Orchestrates parallel retrieval across Graph, Vector, and Admission backends

    according to the configured RETRIEVAL_MODE ('mock', 'hybrid_live', or 'external_api').
    """

    @classmethod
    async def route_initial_retrieval(
        cls,
        patient_id: str,
        plan: InvestigationPlan
    ) -> Tuple[List[AdmissionRecord], List[GraphRecord], List[ClinicalNote]]:
        """Execute parallel retrieval for admissions, graph records, and clinical notes."""
        mode = settings.get_effective_retrieval_mode()
        logger.info(f"Executing retrieval for patient={patient_id} [Mode: {mode}]")

        primary_graph_query = " ".join(plan.graph_queries) if plan.graph_queries else "all records"
        primary_vector_query = " ".join(plan.vector_queries) if plan.vector_queries else "patient history and notes"

        if mode == "hybrid_live":
            # Live native Neo4j and native Vector search
            async def _fetch_admissions():
                return await neo4j_service.get_patient_admissions(patient_id)

            async def _fetch_graph():
                return await neo4j_service.search_patient_graph(patient_id, primary_graph_query)

            async def _fetch_vector():
                return vector_service.search(patient_id, primary_vector_query, top_k=5)

            admissions, graph_records, notes = await asyncio.gather(
                _fetch_admissions(),
                _fetch_graph(),
                _fetch_vector()
            )

        elif mode == "external_api":
            # Remote HTTP API to companion backend
            admissions_task = api_client.get_admissions(patient_id)
            graph_task = api_client.search_graph(patient_id, primary_graph_query)
            vector_task = api_client.search_vector(patient_id, primary_vector_query, top_k=5)

            admissions, graph_records, notes = await asyncio.gather(
                admissions_task,
                graph_task,
                vector_task
            )

        else:
            # Deterministic in-memory synthetic mock
            admissions = MockDataRepository.get_admissions(patient_id)
            graph_records = MockDataRepository.search_graph(patient_id, primary_graph_query)
            notes = MockDataRepository.search_notes(patient_id, primary_vector_query, top_k=5)

        # Security check: ensure strict patient isolation
        admissions = [a for a in admissions if a.patient_id == patient_id]
        graph_records = [g for g in graph_records if g.patient_id == patient_id]
        notes = [n for n in notes if n.patient_id == patient_id]

        return admissions, graph_records, notes

    @classmethod
    async def route_targeted_retrieval(
        cls,
        patient_id: str,
        targeted_queries: List[str]
    ) -> Tuple[List[GraphRecord], List[ClinicalNote]]:
        """Fetch targeted additional records based on missing evidence queries during a retry."""
        mode = settings.get_effective_retrieval_mode()
        logger.info(f"Targeted retrieval for patient {patient_id} [Mode: {mode}] with queries: {targeted_queries}")
        query_str = " ".join(targeted_queries)

        if mode == "hybrid_live":
            async def _fetch_graph():
                return await neo4j_service.search_patient_graph(patient_id, query_str)

            async def _fetch_vector():
                return vector_service.search(patient_id, query_str, top_k=5)

            new_graph, new_notes = await asyncio.gather(_fetch_graph(), _fetch_vector())

        elif mode == "external_api":
            graph_task = api_client.search_graph(patient_id, query_str)
            vector_task = api_client.search_vector(patient_id, query_str, top_k=5)
            new_graph, new_notes = await asyncio.gather(graph_task, vector_task)

        else:
            new_graph = MockDataRepository.search_graph(patient_id, query_str)
            new_notes = MockDataRepository.search_notes(patient_id, query_str, top_k=5)

        # Security check: ensure strict patient isolation
        new_graph = [g for g in new_graph if g.patient_id == patient_id]
        new_notes = [n for n in new_notes if n.patient_id == patient_id]

        return new_graph, new_notes
