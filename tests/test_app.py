from unittest.mock import patch

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_check_endpoint_requires_url():
    response = client.post("/check", json={})

    assert response.status_code == 422


def test_check_endpoint_returns_combined_json():
    with patch("app.check_url_health") as mock_health, patch("app.extract_visible_layer") as mock_page:
        mock_health.return_value = {"health": "OK", "status": 200}
        mock_page.return_value = {"title": "Example", "login_form_present": False}

        response = client.post("/check", json={"url": "https://example.com"})

    assert response.status_code == 200
    assert response.json() == {
        "url": "https://example.com",
        "health_check": {"health": "OK", "status": 200},
        "page_check": {"title": "Example", "login_form_present": False},
    }
