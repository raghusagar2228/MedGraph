"""Neo4j Database Seeder for MedGraph.

Ingests synthetic patient cohorts (P001 - P004) into a running Neo4j instance
with uniqueness constraints, nodes, and clinical relationships.
"""

import logging
from app.services.neo4j_service import neo4j_service
from app.services.mock_data import (
    P001_ADMISSIONS, P001_GRAPH_RECORDS,
    P002_ADMISSIONS, P002_GRAPH_RECORDS,
    P003_ADMISSIONS, P003_GRAPH_RECORDS,
    P004_ADMISSIONS, P004_GRAPH_RECORDS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def seed_neo4j_database() -> int:
    """Connect to Neo4j, create schema constraints, and ingest synthetic patient cohorts."""
    neo4j_service.verify_connectivity()
    driver = neo4j_service.get_driver()

    total_ingested = 0

    with driver.session(database=neo4j_service.database) as session:
        # 1. Create uniqueness constraints
        constraints = [
            "CREATE CONSTRAINT patient_id_unique IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT admission_id_unique IF NOT EXISTS FOR (a:Admission) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT prescription_id_unique IF NOT EXISTS FOR (r:Prescription) REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT lab_id_unique IF NOT EXISTS FOR (l:LabResult) REQUIRE l.id IS UNIQUE",
            "CREATE CONSTRAINT condition_id_unique IF NOT EXISTS FOR (c:Condition) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT allergy_id_unique IF NOT EXISTS FOR (al:Allergy) REQUIRE al.id IS UNIQUE"
        ]
        for c in constraints:
            try:
                session.run(c)
            except Exception as e:
                logger.debug(f"Constraint creation notice: {e}")

        cohorts = [
            ("P001", "Marcus Vance", P001_ADMISSIONS, P001_GRAPH_RECORDS),
            ("P002", "Elena Rostova", P002_ADMISSIONS, P002_GRAPH_RECORDS),
            ("P003", "David Kim", P003_ADMISSIONS, P003_GRAPH_RECORDS),
            ("P004", "Sarah Jenkins", P004_ADMISSIONS, P004_GRAPH_RECORDS),
        ]

        for patient_id, name, admissions, graph_records in cohorts:
            # Upsert Patient
            session.run(
                "MERGE (p:Patient {id: $id}) SET p.name = $name",
                {"id": patient_id, "name": name}
            )
            total_ingested += 1

            # Ingest Admissions
            for adm in admissions:
                session.run(
                    """
                    MERGE (a:Admission {id: $admission_id})
                    SET a.admission_date = $admission_date,
                        a.discharge_date = $discharge_date,
                        a.reason = $reason,
                        a.department = $department,
                        a.attending_physician = $attending_physician
                    WITH a
                    MATCH (p:Patient {id: $patient_id})
                    MERGE (p)-[:HAS_ADMISSION]->(a)
                    """,
                    adm.model_dump()
                )
                total_ingested += 1

            # Ingest Graph Records
            for rec in graph_records:
                if rec.record_type == "medication":
                    session.run(
                        """
                        MERGE (r:Prescription {id: $record_id})
                        SET r.name = $name,
                            r.dosage = $dosage,
                            r.frequency = $frequency,
                            r.timestamp = $timestamp
                        WITH r
                        MATCH (p:Patient {id: $patient_id})
                        MERGE (p)-[:HAS_PRESCRIPTION]->(r)
                        """,
                        {
                            "record_id": rec.record_id,
                            "patient_id": patient_id,
                            "name": rec.details.get("name", "Unknown"),
                            "dosage": rec.details.get("dosage", "Unspecified"),
                            "frequency": rec.details.get("frequency", "Unspecified"),
                            "timestamp": rec.timestamp or ""
                        }
                    )
                    # Relate to admission if specified
                    for rel in rec.relationships:
                        target = rel.get("target")
                        if target and target.startswith("ADM-"):
                            session.run(
                                """
                                MATCH (a:Admission {id: $adm_id}), (r:Prescription {id: $rx_id})
                                MERGE (a)-[:HAS_PRESCRIPTION]->(r)
                                """,
                                {"adm_id": target, "rx_id": rec.record_id}
                            )
                    total_ingested += 1

                elif rec.record_type == "lab_result":
                    session.run(
                        """
                        MERGE (l:LabResult {id: $record_id})
                        SET l.test_name = $test_name,
                            l.value = $value,
                            l.unit = $unit,
                            l.flag = $flag,
                            l.timestamp = $timestamp
                        WITH l
                        MATCH (p:Patient {id: $patient_id})
                        MERGE (p)-[:HAS_LAB_RESULT]->(l)
                        """,
                        {
                            "record_id": rec.record_id,
                            "patient_id": patient_id,
                            "test_name": rec.details.get("test_name", "Unknown"),
                            "value": rec.details.get("value", 0.0),
                            "unit": rec.details.get("unit", ""),
                            "flag": rec.details.get("flag", "NORMAL"),
                            "timestamp": rec.timestamp or ""
                        }
                    )
                    for rel in rec.relationships:
                        target = rel.get("target")
                        if target and target.startswith("ADM-"):
                            session.run(
                                """
                                MATCH (a:Admission {id: $adm_id}), (l:LabResult {id: $lab_id})
                                MERGE (a)-[:HAS_LAB_RESULT]->(l)
                                """,
                                {"adm_id": target, "lab_id": rec.record_id}
                            )
                    total_ingested += 1

                elif rec.record_type == "condition":
                    session.run(
                        """
                        MERGE (c:Condition {id: $record_id})
                        SET c.name = $name,
                            c.icd10 = $icd10,
                            c.status = $status,
                            c.timestamp = $timestamp
                        WITH c
                        MATCH (p:Patient {id: $patient_id})
                        MERGE (p)-[:HAS_CONDITION]->(c)
                        """,
                        {
                            "record_id": rec.record_id,
                            "patient_id": patient_id,
                            "name": rec.details.get("name", "Unknown"),
                            "icd10": rec.details.get("icd10", ""),
                            "status": rec.details.get("status", "Active"),
                            "timestamp": rec.timestamp or ""
                        }
                    )
                    total_ingested += 1

                elif rec.record_type == "allergy":
                    session.run(
                        """
                        MERGE (al:Allergy {id: $record_id})
                        SET al.allergen = $allergen,
                            al.severity = $severity,
                            al.reaction = $reaction,
                            al.timestamp = $timestamp
                        WITH al
                        MATCH (p:Patient {id: $patient_id})
                        MERGE (p)-[:HAS_ALLERGY]->(al)
                        """,
                        {
                            "record_id": rec.record_id,
                            "patient_id": patient_id,
                            "allergen": rec.details.get("allergen", "Unknown"),
                            "severity": rec.details.get("severity", "UNKNOWN"),
                            "reaction": rec.details.get("reaction", ""),
                            "timestamp": rec.timestamp or ""
                        }
                    )
                    total_ingested += 1

    logger.info(f"Successfully seeded {total_ingested} clinical entities into Neo4j.")
    return total_ingested


if __name__ == "__main__":
    try:
        count = seed_neo4j_database()
        print(f"Neo4j seeding complete. Ingested {count} entities.")
    except Exception as e:
        print(f"Seeding failed: {e}")
