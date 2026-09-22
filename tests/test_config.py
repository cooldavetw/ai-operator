import pytest
from pydantic import ValidationError

from app.config import DomainConfig, Settings, load_config
from app.store import Store


def test_rules_cannot_ambiguously_map_to_two_domains(settings):
    value = load_config(settings.config_path).model_dump()
    value["domains"]["ACADEMIC_SYSTEM"]["exact_matches"].append("我要續借書")
    with pytest.raises(ValidationError):
        DomainConfig.model_validate(value)


def test_unknown_reserved_and_two_turn_limit(settings):
    value = load_config(settings.config_path).model_dump()
    value["policy"]["max_clarifications"] = 3
    with pytest.raises(ValidationError):
        DomainConfig.model_validate(value)
    value["policy"]["max_clarifications"] = 2
    value["domains"]["UNKNOWN"] = value["domains"].pop("ACADEMIC_SYSTEM")
    with pytest.raises(ValidationError):
        DomainConfig.model_validate(value)


def test_api_key_required_and_no_example_key():
    for key in ["", "   ", "replace-with-a-random-secret"]:
        with pytest.raises(ValidationError):
            Settings(api_key=key)


@pytest.mark.parametrize("key", ["a", "test", "short"])
def test_short_api_keys_allowed(key):
    assert Settings(api_key=key).api_key.get_secret_value() == key


def test_database_refuses_second_owner(settings):
    first, second = Store(settings.database_path), Store(settings.database_path)
    first.open()
    try:
        with pytest.raises(RuntimeError, match="one API worker"):
            second.open()
    finally:
        first.close()
        second.close()

