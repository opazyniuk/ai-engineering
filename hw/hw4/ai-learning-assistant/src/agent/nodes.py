from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from src.agent.prompts import DECOMPOSE_PROMPT, EXPLAIN_PROMPT
from src.agent.state import AgentState
from src.models.concept import Concept, DecompositionResult, ExplanationResult, KnowledgeGraph


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


def initialize(state: AgentState) -> dict:
    """Create root concept and set it as pending.

    Без LLM. Створює корінь графа з теми ("Backpropagation") і кладе його
    в pending_concept_ids — це ЧЕРГА на розкладання. Decompose далі бере
    елементи з цієї черги один за одним.
    """
    kg = KnowledgeGraph()
    root = Concept(name=state["topic"], level=0)
    kg.add_concept(root)
    kg.root_id = root.id
    return {
        "knowledge_graph": kg,
        "pending_concept_ids": [root.id],
    }


def decompose(state: AgentState) -> dict:
    """Take the first pending concept and decompose it into prerequisites.

    Один виклик = одна гілка. Workflow зациклений (decompose → explain →
    decompose → ...), тому глибокий граф будується по одному рівню за ітерацію.
    """
    kg: KnowledgeGraph = state["knowledge_graph"]
    pending = list(state["pending_concept_ids"])

    if not pending:
        return {"pending_concept_ids": []}

    # FIFO: беремо перший concept з черги і прибираємо з pending.
    concept_id = pending.pop(0)
    concept = kg.concepts[concept_id]

    # Передається в промпт щоб LLM не повертав дублікати.
    existing_names = [c.name.lower() for c in kg.concepts.values()]

    # with_structured_output — змушує LLM повернути JSON за схемою Pydantic
    # (DecompositionResult), не вільний текст. Гарантія типу. Якщо LLM
    # відповість невалідним JSON — Pydantic кине помилку валідації.
    llm = _get_llm().with_structured_output(DecompositionResult)
    prompt = DECOMPOSE_PROMPT.format(
        topic=state["topic"],
        user_level=state["user_level"],
        concept_name=concept.name,
        existing_concepts=", ".join(existing_names),
    )

    result: DecompositionResult = llm.invoke([SystemMessage(content=prompt)])

    # Додаємо знайдені prerequisites у граф.
    new_pending = []
    for prereq_name in result.prerequisites:
        # Дедуп: якщо такий concept вже є — пропускаємо.
        if prereq_name.lower() in existing_names:
            continue
        child = Concept(name=prereq_name, level=concept.level + 1)
        kg.add_concept(child)
        kg.add_edge(concept.id, child.id)
        existing_names.append(prereq_name.lower())
        # ОБМЕЖУВАЧ ГЛИБИНИ: додаємо в граф, але в pending тільки якщо
        # ще не на максимальній глибині. Так workflow знає коли зупинитись.
        if child.level < state["max_depth"]:
            new_pending.append(child.id)

    # Помічаємо що цю гілку розклали — UI ховає кнопку "Expand deeper".
    concept.is_expanded = True

    return {
        "knowledge_graph": kg,
        "current_concept_id": concept_id,
        # Решта pending (без обробленого) + нові діти.
        "pending_concept_ids": pending + new_pending,
    }


def explain(state: AgentState) -> dict:
    """Generate explanations for concepts that don't have one yet."""
    kg: KnowledgeGraph = state["knowledge_graph"]
    llm = _get_llm().with_structured_output(ExplanationResult)

    unexplained = [c for c in kg.concepts.values() if not c.explanation]

    for concept in unexplained:
        # Find parent concept name for context
        parent_name = state["topic"]
        for edge in kg.edges:
            if edge.target == concept.id:
                parent_name = kg.concepts[edge.source].name
                break

        prompt = EXPLAIN_PROMPT.format(
            concept_name=concept.name,
            parent_concept=parent_name,
            user_level=state["user_level"],
        )
        result: ExplanationResult = llm.invoke([SystemMessage(content=prompt)])
        concept.explanation = result.explanation

    return {"knowledge_graph": kg}


def should_continue(state: AgentState) -> str:
    """Decide whether to continue decomposing or stop.

    Без LLM. Звичайний if. Повертає рядок який LangGraph використовує
    в conditional_edges (див. graph.py) щоб обрати наступний вузол.
    """
    pending = state.get("pending_concept_ids", [])
    if pending:
        return "continue"
    return "done"
