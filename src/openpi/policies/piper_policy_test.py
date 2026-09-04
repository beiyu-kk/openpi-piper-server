import numpy as np

from openpi.models import model as _model
from openpi.policies import piper_policy


def test_piper_inputs_maps_cameras_and_keeps_continuous_actions():
    base_image = np.zeros((3, 8, 10), dtype=np.float32)
    wrist_image = np.ones((3, 8, 10), dtype=np.float32)
    actions = np.linspace(0.0, 1.0, 14, dtype=np.float32).reshape(2, 7)
    transform = piper_policy.PiperInputs(model_type=_model.ModelType.PI05)

    output = transform(
        {
            "observation/top_image": base_image,
            "observation/right_wrist_image": wrist_image,
            "observation/state": np.arange(7, dtype=np.float32),
            "actions": actions,
            "prompt": "pick up the book",
        }
    )

    assert output["image"]["base_0_rgb"].shape == (8, 10, 3)
    assert output["image"]["right_wrist_0_rgb"].shape == (8, 10, 3)
    assert not output["image_mask"]["left_wrist_0_rgb"]
    np.testing.assert_array_equal(output["actions"], actions)
    assert output["actions"][..., -1].tolist() == actions[..., -1].tolist()


def test_piper_outputs_returns_seven_continuous_dimensions():
    actions = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(2, 32)
    output = piper_policy.PiperOutputs()({"actions": actions})
    np.testing.assert_array_equal(output["actions"], actions[..., :7])
