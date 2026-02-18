import os
import importlib
import pytest


# -------------------------------------------------
# Helper to reload module after env change
# -------------------------------------------------

def reload_settings_module(monkeypatch, env=None):
    """
    Utility to patch environment variables
    and reload the settings module.
    """
    env = env or {}

    # Clear relevant vars first
    monkeypatch.delenv("AI_ENV", raising=False)
    monkeypatch.delenv("AI_DEBUG", raising=False)

    # Set new ones
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    import rakam_systems_core.config as config
    importlib.reload(config)

    return config


# -------------------------------------------------
# Tests
# -------------------------------------------------

def test_defaults(monkeypatch):
    module = reload_settings_module(monkeypatch)

    settings = module.settings

    assert settings.env == "dev"
    assert settings.debug is False


def test_env_override(monkeypatch):
    module = reload_settings_module(
        monkeypatch,
        env={"AI_ENV": "prod"}
    )

    settings = module.settings

    assert settings.env == "prod"
    assert settings.debug is False


def test_debug_enabled(monkeypatch):
    module = reload_settings_module(
        monkeypatch,
        env={"AI_DEBUG": "1"}
    )

    settings = module.settings

    assert settings.env == "dev"
    assert settings.debug is True


def test_env_and_debug_together(monkeypatch):
    module = reload_settings_module(
        monkeypatch,
        env={
            "AI_ENV": "staging",
            "AI_DEBUG": "1",
        }
    )

    settings = module.settings

    assert settings.env == "staging"
    assert settings.debug is True


def test_debug_not_enabled_for_other_values(monkeypatch):
    module = reload_settings_module(
        monkeypatch,
        env={"AI_DEBUG": "true"}  # only "1" should enable
    )

    settings = module.settings

    assert settings.debug is False
