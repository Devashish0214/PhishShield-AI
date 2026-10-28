from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200

def test_analyze_phishing():
    response = client.post("/api/analyze", json={"content": "Urgent: Your account is suspended. Verify credentials now at http://bit.ly/fake"})
    assert response.status_code == 200
    data = response.json()
    assert data["threat_score"] > 50
    assert "verdict" in data

def test_analyze_safe():
    response = client.post("/api/analyze", json={"content": "Your dental appointment is scheduled for tomorrow at 2:30 PM."})
    assert response.status_code == 200
    data = response.json()
    assert data["threat_score"] <= 50
