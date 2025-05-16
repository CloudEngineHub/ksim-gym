"""Converts a checkpoint to a deployable model."""
from __future__ import annotations
import sys


import argparse
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import ksim
from jaxtyping import Array
from kinfer.export.jax import export_fn
from kinfer.export.serialize import pack

import importlib.util
#  from train import HumanoidWalkingTask, Model # << NOW DONE INLINE


def make_export_model(model: "Model") -> Callable:
    def model_fn(obs: Array, carry: Array) -> tuple[Array, Array]:
        dist, carry = model.actor.forward(obs, carry)
        return dist.mode(), carry

    def batched_model_fn(obs: Array, carry: Array) -> tuple[Array, Array]:
        return jax.vmap(model_fn)(obs, carry)

    return batched_model_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--train_override", type=str, default=None)
    args = parser.parse_args()

    training_path = None
    if args.train_override is None: 
        # Default Path, local train version
        training_path = Path(__file__).parent / "train.py"
    else:
        if not (training_path := Path(args.train_override)).exists():
            raise FileNotFoundError(f"Training code path {training_path} does not exist")
        else:
            sys.path.append(str(training_path.parent))
    try:
        spec = importlib.util.spec_from_file_location(training_path.stem, training_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get Classes
        HumanoidWalkingTask = getattr(module, "HumanoidWalkingTask")
        Model = getattr(module, "Model")
    except Exception as e:
        raise ImportError(f"Failed to import training code from {training_path}: {e}")
    
    if not (ckpt_path := Path(args.checkpoint_path)).exists():
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist")

    # Patch the classes to make inspect.getfile work
    HumanoidWalkingTask.__module__ = "__main__"  # Set the module name
    HumanoidWalkingTask.__file__ = str(training_path)  # Set the file path

    # If needed, do the same for Model
    Model.__module__ = "__main__"
    Model.__file__ = str(training_path)

    task: HumanoidWalkingTask = HumanoidWalkingTask.load_task(ckpt_path)
    model: Model = task.load_ckpt(ckpt_path, part="model")[0]

    # Loads the Mujoco model and gets the joint names.
    mujoco_model = task.get_mujoco_model()
    joint_names = ksim.get_joint_names_in_order(mujoco_model)[1:]  # Removes the root joint.

    # Constant values.
    carry_shape = (task.config.depth, task.config.hidden_size)
    num_joints = len(joint_names)

    @jax.jit
    def init_fn() -> Array:
        return jnp.zeros(carry_shape)

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        projected_gravity: Array,
        accelerometer: Array,
        gyroscope: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        obs = jnp.concatenate(
            [
                joint_angles,
                joint_angular_velocities,
                projected_gravity,
                accelerometer,
                gyroscope,
            ],
            axis=-1,
        )
        dist, carry = model.actor.forward(obs, carry)
        return dist.mode(), carry

    init_onnx = export_fn(
        model=init_fn,
        num_joints=num_joints,
        carry_shape=carry_shape,
    )

    step_onnx = export_fn(
        model=step_fn,
        num_joints=num_joints,
        carry_shape=carry_shape,
    )

    kinfer_model = pack(
        init_fn=init_onnx,
        step_fn=step_onnx,
        joint_names=joint_names,
        carry_shape=carry_shape,
    )

    # Saves the resulting model.
    (output_path := Path(args.output_path)).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(kinfer_model)


if __name__ == "__main__":
    main()
