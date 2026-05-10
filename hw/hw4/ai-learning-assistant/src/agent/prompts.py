DECOMPOSE_PROMPT = """\
You are an expert educator. Given a concept, identify its key prerequisite concepts \
that a student must understand first.

Topic: {topic}
Student level: {user_level}
Current concept to decompose: {concept_name}

Rules:
- Return 2-4 prerequisite concepts that are DIRECT prerequisites.
- Each prerequisite should be more fundamental than the current concept.
- Adapt complexity to the student level:
  - beginner: break down to very basic fundamentals
  - intermediate: assume basic math/programming knowledge
  - advanced: only list non-obvious prerequisites
- Use short, clear concept names (1-4 words).
- Do NOT repeat concepts already in the graph: {existing_concepts}
"""

EXPLAIN_PROMPT = """\
You are an expert educator. Provide a clear, concise explanation of the concept below.

Concept: {concept_name}
Context: This is a prerequisite for understanding "{parent_concept}".
Student level: {user_level}

Rules:
- 2-4 sentences maximum.
- Use simple language appropriate for the student level.
- Include one concrete example or analogy if possible.
- Focus on intuition, not formal definitions.
"""
