from __future__ import annotations

from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.models.concept import KnowledgeGraph


class AgentState(TypedDict):
    topic: str
    user_level: str  # beginner / intermediate / advanced
    max_depth: int
    knowledge_graph: KnowledgeGraph
    current_concept_id: str
    pending_concept_ids: list[str]
    messages: Annotated[list, add_messages]
