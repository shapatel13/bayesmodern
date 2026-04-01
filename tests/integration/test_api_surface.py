from fastapi.testclient import TestClient

from apps.api.main import app


def test_analysis_endpoint_returns_research_report() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/analyze",
        json={
            "case_id": "api-pe-1",
            "note_text": "Pleuritic chest pain with tachycardia and hypoxemia after immobility.",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["differential"]["ranked"][0]["slug"] == "pe"

