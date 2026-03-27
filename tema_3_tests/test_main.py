import requests
import httpx
import sys
import pytest
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from tema_3_evaluation.groq_llm import GroqDeepEval
from dotenv import load_dotenv

load_dotenv()

# foloseste UTF-8 pentru stdout ca sa evite erori de codare
sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://localhost:8000"

groq_model = GroqDeepEval()

relevance_metric = GEval(
    name="Relevanta Fitness",
    criteria=(
        "Raspunsul trebuie sa fie relevant pentru domeniul fitness, exercitii fizice, "
        "antrenamente sau nutritie sportiva. Verifica daca raspunsul ofera informatii utile "
        "si corecte legate de intrebarea pusa."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=groq_model,
    threshold=0.7,
)

refusal_metric = GEval(
    name="Refuz Corect",
    criteria=(
        "Raspunsul trebuie sa refuze politicos sa raspunda la intrebari care nu sunt legate "
        "de fitness, exercitii, antrenamente sau nutritie, si sa redirectioneze utilizatorul "
        "catre subiecte relevante de fitness."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=groq_model,
    threshold=0.7,
)


def test_root_endpoint():
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


def test_chat_relevant_question():
    """Test pozitiv: intrebare relevanta despre fitness evaluata de LLM as a Judge."""
    user_input = "Ce exercitii pot face acasa pentru a slabi si a-mi imbunatati conditia fizica?"
    response = requests.post(f"{BASE_URL}/chat/", json={"message": user_input}, timeout=60)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data

    test_case = LLMTestCase(
        input=user_input,
        actual_output=data["response"],
    )
    relevance_metric.measure(test_case)
    assert relevance_metric.score >= 0.7, (
        f"Scor relevanta prea mic: {relevance_metric.score:.2f} — {relevance_metric.reason}"
    )


def test_chat_irrelevant_question():
    """Test negativ: intrebare irelevanta trebuie refuzata politicos, evaluata de LLM as a Judge."""
    user_input = "Care este capitala Frantei si ce moneda folosesc?"
    response = requests.post(f"{BASE_URL}/chat/", json={"message": user_input}, timeout=60)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data

    test_case = LLMTestCase(
        input=user_input,
        actual_output=data["response"],
    )
    refusal_metric.measure(test_case)
    assert refusal_metric.score >= 0.7, (
        f"Scor refuz prea mic: {refusal_metric.score:.2f} — {refusal_metric.reason}"
    )
