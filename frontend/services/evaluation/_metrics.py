from typing_extensions import Annotated, TypedDict

from models import (
    llm,
    CorrectnessGrade,
    RelevanceGrade,
    GroundedGrade
)
from ._prompts import (
    correctness_prompt,
    relevance_prompt,
    grounded_prompt
)


evaluator_llm = llm()


def correctness(inputs: dict,
                outputs: dict,
                reference_outputs: dict) -> bool:
    """An evaluator for RAG answer accuracy
    """

    answers = f"""\
        QUESTION: {inputs['question']}
        GROUND TRUTH ANSWER: {reference_outputs['answer']}
        STUDENT ANSWER: {outputs['answer']}
    """

    correctness_llm = evaluator_llm.with_structured_output(
        CorrectnessGrade,
        method="json_schema",
        strict=True
    )

    grade = correctness_llm.invoke([
        {"role": "system", "content": correctness_prompt},
        {"role": "user", "content": answers}
    ])

    return grade["correct"]

def relevance(inputs: dict, outputs: dict) -> bool:
    """A simple evaluator for RAG answer helpfulness
    """

    answer = f"""QUESTION: {inputs['question']}
        STUDENT ANSWER: {outputs['answer']}
    """

    relevance_llm = evaluator_llm.with_structured_output(
        RelevanceGrade,
        method="json_schema",
        strict=True
    )

    grade = relevance_llm([
        {"role": "system", "content": relevance_prompt},
        {"role": "user", "content": answer}
    ])

    return grade["relevant"]

def groundedness(inputs: dict, outputs: dict) -> bool:
    """A simple evaluator for RAG answer groundness
    """

    grounded_llm = evaluator_llm.with_structured_output(
        GroundedGrade,
        method="json_schema",
        strict=True
    )

    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"

    grade = grounded_llm.invoke([
        {"role": "system", "content": grounded_prompt},
        {"role": "user", "content": answer}
    ])

    return grade["grounded"]
    