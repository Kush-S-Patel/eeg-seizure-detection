from streamlit.testing.v1 import AppTest


def test_dashboard_starts():
    app = AppTest.from_file("app.py")
    app.run(timeout=60)
    assert not app.exception
