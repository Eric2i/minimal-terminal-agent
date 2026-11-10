import pytest

pytest.importorskip("daytona")


def test_daytona_environment_defaults_to_auto_stop(monkeypatch):
    """Harbor's sandboxes never auto-stop by default; ours do, unless overridden."""
    from harbor.environments.daytona import DaytonaEnvironment as Base

    from mta.integrations.harbor.daytona import DEFAULT_AUTO_STOP_MINS, DaytonaEnvironment

    seen = []
    monkeypatch.setattr(Base, "__init__", lambda self, *a, **kw: seen.append(kw["auto_stop_interval_mins"]))
    DaytonaEnvironment()
    DaytonaEnvironment(auto_stop_interval_mins=15)
    assert seen == [DEFAULT_AUTO_STOP_MINS, 15]
