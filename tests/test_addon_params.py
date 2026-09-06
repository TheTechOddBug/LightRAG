"""Tests for observable runtime addon parameters."""

import pytest

from lightrag.addon_params import ObservableAddonParams

pytestmark = pytest.mark.offline


def test_update_commits_valid_input_and_notifies_once() -> None:
    changes: list[str] = []
    params = ObservableAddonParams(
        {"language": "English"}, on_change=lambda: changes.append("changed")
    )

    params.update([("language", "French")], entity_types=["person"])

    assert params == {"language": "French", "entity_types": ["person"]}
    assert changes == ["changed"]


def test_update_is_atomic_when_an_iterable_is_malformed() -> None:
    changes: list[str] = []

    def unexpected_change() -> None:
        changes.append("changed")
        raise AssertionError("failed updates must not trigger callbacks")

    params = ObservableAddonParams({"language": "English"}, on_change=unexpected_change)

    def partly_invalid_items():
        yield ("language", "French")
        yield ("malformed",)

    with pytest.raises(ValueError, match="dictionary update sequence element #1"):
        params.update(partly_invalid_items())

    # A failed live-config update must not leave the mapping ahead of its
    # derived cache or emit a change notification for an update that failed.
    assert params == {"language": "English"}
    assert changes == []


def test_update_does_not_notify_when_validation_fails_before_a_write() -> None:
    changes: list[str] = []
    params = ObservableAddonParams(
        {"language": "English"}, on_change=lambda: changes.append("changed")
    )

    with pytest.raises(TypeError):
        params.update(42)

    assert params == {"language": "English"}
    assert changes == []
