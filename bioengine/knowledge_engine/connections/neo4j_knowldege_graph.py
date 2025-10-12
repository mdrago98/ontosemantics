from typing import Dict

from bioengine.knowledge_engine.connections.connection import KnowledgeStore
from bioengine.knowledge_engine.models.relationship import VerifiedRelationship
from settings import Config


class Neo4jKnowledgeStore(KnowledgeStore):
    """Neo4j implementation of knowledge storage"""

    def __init__(self, config: Config):
        self.driver = self._connect(config)

    def store_relationship(self, relationship: VerifiedRelationship) -> bool:
        # Neo4j storage logic
        pass
