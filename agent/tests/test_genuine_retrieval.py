import os
import pytest
import numpy as np
from app.config import Settings
from app.schemas.evidence import ClinicalNote
from app.services.vector_service import VectorService
from app.services.neo4j_service import Neo4jService


@pytest.fixture
def temp_vector_service(tmp_path):
    """Isolated vector service using a temporary directory for tests."""
    test_index_path = str(tmp_path / "test_vector_index.json")
    service = VectorService(index_path=test_index_path)
    return service


def test_vector_embedding_generation(temp_vector_service):
    """Verify that FastEmbed generates normalized 384-dimensional dense vectors."""
    text = "Patient presenting with acute heart failure exacerbation."
    embedding = temp_vector_service.generate_embedding(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)
    # Normalized vector should have L2 norm close to 1.0
    norm = float(np.linalg.norm(embedding))
    assert abs(norm - 1.0) < 1e-4


def test_vector_indexing_and_persistence(temp_vector_service):
    """Verify that clinical notes can be embedded, stored, and reloaded from disk."""
    note = ClinicalNote(
        record_id="NOTE-TEST-01",
        patient_id="P_TEST",
        timestamp="2026-01-01T00:00:00Z",
        author="Dr. Test",
        note_type="consultation",
        content="Patient initiated on Metoprolol Succinate 50mg daily."
    )

    count = temp_vector_service.index_notes([note])
    assert count == 1
    assert temp_vector_service.count() == 1
    assert os.path.exists(temp_vector_service.index_path)

    # Reload from disk in a fresh instance
    reloaded_service = VectorService(index_path=temp_vector_service.index_path)
    assert reloaded_service.count() == 1


def test_vector_semantic_search_ranking(temp_vector_service):
    """Verify that cosine similarity ranking prioritizes clinically matching concepts."""
    notes = [
        ClinicalNote(
            record_id="NOTE-CARDIAC",
            patient_id="P_CARDIAC",
            timestamp="2026-02-01T10:00:00Z",
            content="Cardiology note: Severe congestive heart failure with shortness of breath and peripheral edema. Prescribed Entresto."
        ),
        ClinicalNote(
            record_id="NOTE-DERM",
            patient_id="P_CARDIAC",
            timestamp="2026-02-01T11:00:00Z",
            content="Dermatology evaluation: Mild skin rash on left arm resolved with topical hydrocortisone cream."
        )
    ]
    temp_vector_service.index_notes(notes)

    # Semantic query without exact word overlap
    query = "cardiac dyspnea fluid overload heart failure medication"
    results = temp_vector_service.search(patient_id="P_CARDIAC", query=query, top_k=2)

    assert len(results) == 2
    # The cardiac note must rank first with higher cosine similarity
    assert results[0].record_id == "NOTE-CARDIAC"
    assert results[0].relevance_score > results[1].relevance_score
    assert results[0].source == "vector_native"


def test_vector_patient_isolation(temp_vector_service):
    """Verify strict patient pre-filtering: records from patient A never leak to patient B."""
    notes = [
        ClinicalNote(
            record_id="NOTE-PATIENT-A",
            patient_id="PATIENT_A",
            content="Patient A critical heart failure notes."
        ),
        ClinicalNote(
            record_id="NOTE-PATIENT-B",
            patient_id="PATIENT_B",
            content="Patient B critical heart failure notes."
        )
    ]
    temp_vector_service.index_notes(notes)

    # Query for Patient A should NEVER return Patient B's note
    res_a = temp_vector_service.search(patient_id="PATIENT_A", query="critical heart failure", top_k=5)
    assert len(res_a) == 1
    assert res_a[0].record_id == "NOTE-PATIENT-A"
    assert res_a[0].patient_id == "PATIENT_A"


def test_neo4j_explicit_failure_when_offline():
    """Verify that when Neo4j is offline, verify_connectivity() raises an explicit

    ConnectionError with diagnostics rather than silently masking with mock data.
    """
    offline_service = Neo4jService(
        uri="bolt://localhost:19999",  # Non-existent port
        user="neo4j",
        password="bad_password"
    )

    with pytest.raises(ConnectionError, match="Cannot connect to Neo4j database"):
        offline_service.verify_connectivity()


def test_config_explicit_retrieval_modes():
    """Verify that Settings resolves modes explicitly and prevents silent ambiguity."""
    # Explicit mock
    s_mock = Settings(retrieval_mode="mock")
    assert s_mock.get_effective_retrieval_mode() == "mock"

    # Explicit hybrid live
    s_live = Settings(retrieval_mode="hybrid_live")
    assert s_live.get_effective_retrieval_mode() == "hybrid_live"

    # Explicit external api
    s_api = Settings(retrieval_mode="external_api")
    assert s_api.get_effective_retrieval_mode() == "external_api"


@pytest.mark.asyncio
async def test_workflow_hybrid_live_explicit_neo4j_diagnostic(monkeypatch):
    """Verify that when RETRIEVAL_MODE is set to 'hybrid_live' and Neo4j is offline,

    the workflow raises an explicit ConnectionError instead of silently faking data.
    """
    from app.main import run_investigation
    from app.config import settings

    monkeypatch.setenv("RETRIEVAL_MODE", "hybrid_live")
    monkeypatch.setattr(settings, "retrieval_mode", "hybrid_live")

    # Offline Neo4j must raise ConnectionError during retrieval
    with pytest.raises(ConnectionError, match="Cannot connect to Neo4j database"):
        await run_investigation(
            patient_id="P001",
            user_question="Medications in latest visit?",
            thread_id="test_hybrid_live_fail"
        )
