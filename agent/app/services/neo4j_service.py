import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError

from app.config import settings
from app.schemas.evidence import GraphRecord, AdmissionRecord

logger = logging.getLogger(__name__)


class Neo4jService:
    """Official Neo4j Bolt driver service executing real parameterized Cypher queries

    against the clinical knowledge graph with explicit connection diagnostics.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None
    ):
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self.database = database or settings.neo4j_database
        self._driver: Optional[Driver] = None

    def get_driver(self) -> Driver:
        """Get or initialize the Neo4j driver with configured connection timeout."""
        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password),
                    connection_timeout=settings.neo4j_connection_timeout
                )
            except Exception as e:
                logger.error(f"Failed to create Neo4j driver for {self.uri}: {e}")
                raise ConnectionError(
                    f"Failed to initialize Neo4j driver for '{self.uri}'. "
                    f"Please verify NEO4J_URI in your configuration: {e}"
                )
        return self._driver

    def verify_connectivity(self) -> bool:
        """Verify that the Neo4j server is alive and credentials are valid.

        Raises ConnectionError with explicit diagnostics if unavailable.
        """
        driver = self.get_driver()
        try:
            driver.verify_connectivity()
            return True
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable at {self.uri}: {e}")
            raise ConnectionError(
                f"Cannot connect to Neo4j database at '{self.uri}'. "
                f"Please ensure the Neo4j service is running (e.g. `docker compose up -d neo4j`): {e}"
            )
        except AuthError as e:
            logger.error(f"Neo4j authentication failed for user '{self.user}': {e}")
            raise ConnectionError(
                f"Neo4j authentication failed for user '{self.user}'. "
                f"Please verify NEO4J_USER and NEO4J_PASSWORD settings: {e}"
            )

    async def search_patient_graph(self, patient_id: str, query: str) -> List[GraphRecord]:
        """Execute parameterized Cypher query to retrieve connected clinical entities

        for a patient (admissions, medications, lab results, conditions, allergies).
        """
        self.verify_connectivity()
        driver = self.get_driver()

        records: List[GraphRecord] = []
        cypher_query = """
        MATCH (p:Patient {id: $patient_id})
        OPTIONAL MATCH (p)-[:HAS_ADMISSION]->(a:Admission)
        OPTIONAL MATCH (a)-[:HAS_PRESCRIPTION]->(rx:Prescription)
        OPTIONAL MATCH (a)-[:HAS_LAB_RESULT]->(lr:LabResult)
        OPTIONAL MATCH (p)-[:HAS_CONDITION]->(c:Condition)
        OPTIONAL MATCH (p)-[:HAS_ALLERGY]->(al:Allergy)
        RETURN a, rx, lr, c, al
        """

        with driver.session(database=self.database) as session:
            result = session.run(cypher_query, {"patient_id": patient_id})
            seen_ids = set()

            for row in result:
                # 1. Admission records
                a_node = row.get("a")
                if a_node and a_node.get("id") and a_node["id"] not in seen_ids:
                    seen_ids.add(a_node["id"])
                    records.append(
                        GraphRecord(
                            record_id=a_node["id"],
                            patient_id=patient_id,
                            record_type="admission",
                            timestamp=a_node.get("admission_date"),
                            details=dict(a_node),
                            relationships=[],
                            source="neo4j_native"
                        )
                    )

                # 2. Prescription records
                rx_node = row.get("rx")
                if rx_node and rx_node.get("id") and rx_node["id"] not in seen_ids:
                    seen_ids.add(rx_node["id"])
                    relationships = []
                    if a_node and a_node.get("id"):
                        relationships.append({"type": "PRESCRIBED_DURING", "target": a_node["id"]})
                    records.append(
                        GraphRecord(
                            record_id=rx_node["id"],
                            patient_id=patient_id,
                            record_type="medication",
                            timestamp=rx_node.get("timestamp"),
                            details=dict(rx_node),
                            relationships=relationships,
                            source="neo4j_native"
                        )
                    )

                # 3. Lab results
                lr_node = row.get("lr")
                if lr_node and lr_node.get("id") and lr_node["id"] not in seen_ids:
                    seen_ids.add(lr_node["id"])
                    relationships = []
                    if a_node and a_node.get("id"):
                        relationships.append({"type": "ORDERED_DURING", "target": a_node["id"]})
                    records.append(
                        GraphRecord(
                            record_id=lr_node["id"],
                            patient_id=patient_id,
                            record_type="lab_result",
                            timestamp=lr_node.get("timestamp"),
                            details=dict(lr_node),
                            relationships=relationships,
                            source="neo4j_native"
                        )
                    )

                # 4. Conditions
                c_node = row.get("c")
                if c_node and c_node.get("id") and c_node["id"] not in seen_ids:
                    seen_ids.add(c_node["id"])
                    records.append(
                        GraphRecord(
                            record_id=c_node["id"],
                            patient_id=patient_id,
                            record_type="condition",
                            timestamp=c_node.get("timestamp"),
                            details=dict(c_node),
                            relationships=[],
                            source="neo4j_native"
                        )
                    )

                # 5. Allergies
                al_node = row.get("al")
                if al_node and al_node.get("id") and al_node["id"] not in seen_ids:
                    seen_ids.add(al_node["id"])
                    records.append(
                        GraphRecord(
                            record_id=al_node["id"],
                            patient_id=patient_id,
                            record_type="allergy",
                            timestamp=al_node.get("timestamp"),
                            details=dict(al_node),
                            relationships=[{"type": "RECORDED_FOR", "target": patient_id}],
                            source="neo4j_native"
                        )
                    )

        logger.info(f"Neo4j search for patient={patient_id} returned {len(records)} records.")
        return records

    async def get_patient_admissions(self, patient_id: str) -> List[AdmissionRecord]:
        """Query chronological hospital admissions for a patient from Neo4j."""
        self.verify_connectivity()
        driver = self.get_driver()

        cypher_query = """
        MATCH (p:Patient {id: $patient_id})-[:HAS_ADMISSION]->(a:Admission)
        RETURN a.id AS admission_id,
               $patient_id AS patient_id,
               a.admission_date AS admission_date,
               a.discharge_date AS discharge_date,
               a.reason AS reason,
               a.department AS department,
               a.attending_physician AS attending_physician
        ORDER BY a.admission_date ASC
        """

        admissions: List[AdmissionRecord] = []
        with driver.session(database=self.database) as session:
            result = session.run(cypher_query, {"patient_id": patient_id})
            for record in result:
                admissions.append(
                    AdmissionRecord(
                        admission_id=record["admission_id"],
                        patient_id=record["patient_id"],
                        admission_date=record["admission_date"],
                        discharge_date=record.get("discharge_date"),
                        reason=record.get("reason"),
                        department=record.get("department"),
                        attending_physician=record.get("attending_physician")
                    )
                )

        logger.info(f"Neo4j admissions lookup for patient={patient_id} returned {len(admissions)} records.")
        return admissions

    def close(self) -> None:
        """Close driver connections."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None


# Global singleton instance
neo4j_service = Neo4jService()
