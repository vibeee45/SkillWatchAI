"""Multipage and Static Branding Assets Test Suite.

Verifies:
1. Root `/` returns HTTP 200 with text/html containing:
   - Official SkillWatchAI branding and logo img (`/logo.png`)
   - Header status pill and navigation tabs (`#navTabs`)
   - All 7 independent page views:
     * #view-live (Live Monitoring & Stream)
     * #view-attendance (Automated Attendance & Audits)
     * #view-students (Student Roster & Face Gallery)
     * #view-occupancy (Spatial & Classroom Occupancy)
     * #view-infrastructure (Equipment & Compliance Checks)
     * #view-compliance (Discrepancy Alerts & Officer Audit)
     * #view-system (Watchdog Health & Offline Queue)
   - Client-side multipage hash router script
2. Branding static endpoints:
   - `/logo.png` returns HTTP 200 with image/png
   - `/favicon.ico` returns HTTP 200 with image/png
   - `/assets/logo.png` returns HTTP 200 with image/png
"""

import pytest
from starlette.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_serves_multipage_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    html = response.text

    # Verify logo presence and branding
    assert 'src="/logo.png"' in html
    assert "SkillWatch AI" in html
    assert "Training Centre Compliance Intelligence" in html

    # Verify navigation tabs container
    assert 'id="navTabs"' in html
    assert 'data-view="live"' in html
    assert 'data-view="attendance"' in html
    assert 'data-view="students"' in html
    assert 'data-view="occupancy"' in html
    assert 'data-view="infrastructure"' in html
    assert 'data-view="compliance"' in html
    assert 'data-view="system"' in html

    # Verify all 7 dedicated view sections exist
    expected_views = [
        'id="view-live"',
        'id="view-attendance"',
        'id="view-students"',
        'id="view-occupancy"',
        'id="view-infrastructure"',
        'id="view-compliance"',
        'id="view-system"',
    ]
    for view_id in expected_views:
        assert view_id in html, f"Missing view section: {view_id}"

    # Verify hash router logic is present
    assert "validViews" in html
    assert "window.location.hash" in html

    # Verify professional enterprise SVG icons replace emojis
    assert 'class="nav-svg"' in html
    assert '<svg class="nav-svg"' in html
    assert "🖥️" not in html
    assert "📋" not in html
    assert "🎓" not in html
    assert "👥" not in html
    assert "🪑" not in html
    assert "⚠️" not in html
    assert "⚙️" not in html


def test_dashboard_summary_infrastructure_counts(client):
    res = client.get("/api/dashboard/summary")
    assert res.status_code == 200
    data = res.json()
    assert "infrastructure" in data
    infra = data["infrastructure"]
    assert "detected_counts" in infra
    assert isinstance(infra["detected_counts"], dict)
    assert "detection_valid" in infra
    assert "detection_state" in infra


def test_branding_logo_endpoints(client):
    # Test top-level /logo.png
    res_logo = client.get("/logo.png")
    assert res_logo.status_code == 200
    assert "image/png" in res_logo.headers.get("content-type", "")
    assert len(res_logo.content) > 1000

    # Test top-level /favicon.ico
    res_fav = client.get("/favicon.ico")
    assert res_fav.status_code == 200
    assert "image/png" in res_fav.headers.get("content-type", "")
    assert len(res_fav.content) > 1000

    # Test static assets directory mount /assets/logo.png
    res_asset = client.get("/assets/logo.png")
    assert res_asset.status_code == 200
    assert "image/png" in res_asset.headers.get("content-type", "")
    assert len(res_asset.content) == len(res_logo.content)
