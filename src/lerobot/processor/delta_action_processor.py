# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

from torch import Tensor

from lerobot.configs.types import FeatureType, PolicyFeature

from .pipeline import ActionProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("map_tensor_to_delta_action_dict")
@dataclass
class MapTensorToDeltaActionDictStep(ActionProcessorStep):
    """
    Map a tensor to a delta action dictionary.
    """

    use_gripper: bool = True

    def action(self, action: Tensor) -> dict:
        if action.dim() > 1:
            action = action.squeeze(0)

        # TODO (maractingi): add rotation
        delta_action = {
            "delta_x": action[0],
            "delta_y": action[1],
            "delta_z": action[2],
        }
        if self.use_gripper:
            delta_action["gripper"] = action[3]
        return delta_action

    def transform_features(
        self, features: dict[FeatureType, dict[str, PolicyFeature]]
    ) -> dict[FeatureType, dict[str, PolicyFeature]]:
        features[FeatureType.ACTION]["delta_x"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        features[FeatureType.ACTION]["delta_y"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        features[FeatureType.ACTION]["delta_z"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        if self.use_gripper:
            features[FeatureType.ACTION]["gripper"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        return features


@ProcessorStepRegistry.register("map_delta_action_to_robot_action")
@dataclass
class MapDeltaActionToRobotActionStep(ActionProcessorStep):
    """
    Map delta actions from teleoperators (gamepad, keyboard) to robot target actions
    for use with inverse kinematics processors.

    Expected input ACTION keys:
    {
        "delta_x": float,
        "delta_y": float,
        "delta_z": float,
        "gripper": float (optional),
    }

    Output ACTION keys:
    {
        "enabled": bool,
        "target_x": float,
        "target_y": float,
        "target_z": float,
        "target_wx": float,
        "target_wy": float,
        "target_wz": float,
        "gripper": float,
    }
    """

    # Scale factors for delta movements
    position_scale: float = 1.0
    rotation_scale: float = 0.0  # No rotation deltas for gamepad/keyboard
    noise_threshold: float = 1e-3  # 1 mm threshold to filter out noise

    def action(self, action: dict) -> dict:
        # NOTE (maractingi): Action can be a dict from the teleop_devices or a tensor from the policy
        # TODO (maractingi): changing this target_xyz naming convention from the teleop_devices
        delta_x = action.pop("delta_x", 0.0)
        delta_y = action.pop("delta_y", 0.0)
        delta_z = action.pop("delta_z", 0.0)
        gripper = action.pop("gripper", 1.0)  # Default to "stay" (1.0)

        # Determine if the teleoperator is actively providing input
        # Consider enabled if any significant movement delta is detected
        position_magnitude = (delta_x**2 + delta_y**2 + delta_z**2) ** 0.5  # Use Euclidean norm for position
        enabled = position_magnitude > self.noise_threshold  # Small threshold to avoid noise

        # Scale the deltas appropriately
        scaled_delta_x = delta_x * self.position_scale
        scaled_delta_y = delta_y * self.position_scale
        scaled_delta_z = delta_z * self.position_scale

        # For gamepad/keyboard, we don't have rotation input, so set to 0
        # These could be extended in the future for more sophisticated teleoperators
        target_wx = 0.0
        target_wy = 0.0
        target_wz = 0.0

        # Update action with robot target format
        action = {
            "enabled": enabled,
            "target_x": scaled_delta_x,
            "target_y": scaled_delta_y,
            "target_z": scaled_delta_z,
            "target_wx": target_wx,
            "target_wy": target_wy,
            "target_wz": target_wz,
            "gripper": float(gripper),
        }

        return action

    def transform_features(
        self, features: dict[FeatureType, dict[str, PolicyFeature]]
    ) -> dict[FeatureType, dict[str, PolicyFeature]]:
        """Transform features to match output format."""
        features[FeatureType.ACTION].pop("delta_x", None)
        features[FeatureType.ACTION].pop("delta_y", None)
        features[FeatureType.ACTION].pop("delta_z", None)
        features[FeatureType.ACTION].pop("gripper", None)

        features[FeatureType.ACTION]["enabled"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        features[FeatureType.ACTION]["target_x"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        features[FeatureType.ACTION]["target_y"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        features[FeatureType.ACTION]["target_z"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        features[FeatureType.ACTION]["target_wx"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        features[FeatureType.ACTION]["target_wy"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        features[FeatureType.ACTION]["target_wz"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        features[FeatureType.ACTION]["gripper"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        return features
