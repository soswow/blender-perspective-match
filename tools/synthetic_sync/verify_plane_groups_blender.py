"""Check native plane-bucket counts and enum persistence in generated Blender data."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile

import bpy


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def register_extension():
    spec = importlib.util.spec_from_file_location(
        "match_perspective", ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.register()
    return module


def labels(properties, landmark):
    # A None context also occurs when Blender asks for enum items outside a UI.
    items = properties._counted_plane_group_items(landmark, None)
    assert [item[0] for item in items] == [str(index) for index in range(1, 11)]
    assert [item[3] for item in items] == list(range(10))
    return [item[1] for item in items]


def add(space, name, *, axis="NONE", group="1", kind="POINT", enabled=True):
    landmark = space.landmarks.add()
    landmark.name = name
    landmark.kind = kind
    landmark.plane_axis = axis
    landmark.plane_group = group
    landmark.use_in_sync = enabled
    return landmark


def run(properties):
    scene = bpy.context.scene
    scene.name = "Plane bucket primary"
    space = scene.match_perspective
    first = space.landmarks.add()
    first.name = "First"
    assert first.plane_group == "1" and first.get("plane_group") is None
    assert labels(properties, first)[0] == "#1 (empty)"
    first.plane_axis = "X"
    assert labels(properties, first)[0] == "#1 (1)"  # Count the edited landmark.

    line = add(space, "Disabled line", axis="X", kind="LINE", enabled=False)
    fourth = add(space, "Fourth", axis="X", group="4")
    other_axis = add(space, "Y", axis="Y")
    no_plane = add(space, "No plane", axis="NONE", group="4")
    # Collection growth may move Blender's RNA elements; reacquire by index.
    first, line, fourth, other_axis, no_plane = space.landmarks
    assert labels(properties, first)[0] == "#1 (2)"
    assert labels(properties, first)[3] == "#4 (1)"
    assert labels(properties, other_axis)[0] == "#1 (1)"
    assert labels(properties, no_plane)[3] == "#4 (empty)"
    assert fourth.get("plane_group") == 3
    assert first.get("plane_group") is None  # Default remains implicit.

    line.plane_group = "2"
    assert labels(properties, first)[0] == "#1 (1)"
    assert labels(properties, first)[1] == "#2 (1)"
    first.plane_group = "4"
    assert labels(properties, first)[0] == "#1 (empty)"
    assert labels(properties, first)[3] == "#4 (2)"
    space.landmarks.remove(2)  # Remove the other X #4 member.
    first = space.landmarks[0]
    assert labels(properties, first)[3] == "#4 (1)"
    assert first.plane_group == "4" and first.get("plane_group") == 3

    second_scene = bpy.data.scenes.new("Plane bucket secondary")
    second = add(second_scene.match_perspective, "Elsewhere", axis="X", group="4")
    assert bpy.context.scene == scene
    assert labels(properties, second)[3] == "#4 (1)"
    assert labels(properties, first)[3] == "#4 (1)"
    second_scene.match_perspective.landmarks.add().plane_axis = "X"
    second = second_scene.match_perspective.landmarks[0]
    assert labels(properties, second)[0] == "#1 (1)"
    assert labels(properties, first)[0] == "#1 (empty)"

    with tempfile.TemporaryDirectory(prefix="pm_plane_groups_") as temp:
        path = str(Path(temp) / "generated.blend")
        bpy.ops.wm.save_as_mainfile(filepath=path)
        bpy.ops.wm.open_mainfile(filepath=path)
        space = bpy.data.scenes["Plane bucket primary"].match_perspective
        first = space.landmarks[0]
        line = space.landmarks[1]
        second = bpy.data.scenes["Plane bucket secondary"].match_perspective.landmarks[0]
        assert first.plane_group == "4" and first.get("plane_group") == 3
        assert line.plane_group == "2" and line.get("plane_group") == 1
        assert second.plane_group == "4" and second.get("plane_group") == 3
        assert labels(properties, first)[3] == "#4 (1)"
        assert labels(properties, second)[0] == "#1 (1)"


if __name__ == "__main__":
    assert not bpy.data.filepath, "Run from --factory-startup with no user file"
    extension = register_extension()
    run(extension.properties)
    print("Plane bucket Blender PASS")
