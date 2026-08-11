from __future__ import annotations

import pytest

from judo_isaaclab.dataset_aliases import (
    canonicalize_named_mapping,
    canonicalize_rigid_object_state,
    parse_object_aliases,
)


def test_legacy_object_labels_are_canonicalized_without_mutation():
    raw = {"obj_0": "mug_asset", "obj_1": "tree_asset"}
    aliases = parse_object_aliases(["obj_0=mug", "obj_1=mug_tree"])
    assert canonicalize_named_mapping(
        raw, aliases, expected_names={"mug", "mug_tree"}
    ) == {"mug": "mug_asset", "mug_tree": "tree_asset"}
    assert raw == {"obj_0": "mug_asset", "obj_1": "tree_asset"}


def test_alias_collisions_fail_closed():
    with pytest.raises(ValueError, match="collide"):
        canonicalize_named_mapping(
            {"obj_0": 1, "mug": 2}, {"obj_0": "mug"}
        )


def test_only_rigid_object_state_keys_are_renamed():
    state = {
        "rigid_object": {"obj_0": "mug pose", "obj_1": "tree pose"},
        "articulation": {"obj_0": "robot joint"},
    }
    canonical = canonicalize_rigid_object_state(
        state, {"obj_0": "mug", "obj_1": "mug_tree"}
    )
    assert canonical["rigid_object"] == {"mug": "mug pose", "mug_tree": "tree pose"}
    assert canonical["articulation"] == {"obj_0": "robot joint"}
    assert state["rigid_object"] == {"obj_0": "mug pose", "obj_1": "tree pose"}
