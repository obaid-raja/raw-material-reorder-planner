from streamlit.testing.v1 import AppTest


def test_app_runs_for_every_material():
    at = AppTest.from_file("../app.py", default_timeout=120).run()
    assert not at.exception
    for key in ("maize", "soybean_meal", "wheat"):
        at.selectbox[0].set_value(key).run()
        assert not at.exception
        assert len(at.metric) == 12
