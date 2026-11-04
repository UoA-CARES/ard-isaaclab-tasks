# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms that do the same randomization as IsaacLab's, but faster."""

from __future__ import annotations

import torch

import isaaclab.envs.mdp as mdp


class randomize_rigid_body_material_cached(mdp.randomize_rigid_body_material):
    """IsaacLab's material randomization, without reading the materials of every env on each call.

    The parent reads all envs' materials on every call, even when only a few envs are due. With 8192
    envs some env is due on nearly every step, so the read runs every step: about 20 ms per call for a
    24-shape hand, against about 1 ms for the write. This term reads once, then keeps the buffer.

    Sampling is the same as the parent's. All terms on one asset share one buffer, because each write
    covers every shape of an env: a separate copy per term would undo the other terms' materials.
    Do not mix this term with the parent, or with other material writes, on the same asset.
    """

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg,
        make_consistent: bool = False,
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        # randomly assign material IDs to the geometries
        view = self.asset.root_physx_view
        bucket_ids = torch.randint(0, num_buckets, (len(env_ids), view.max_shapes), device="cpu")
        material_samples = self.material_buckets[bucket_ids]

        # the buffer shared by all terms on this asset
        if not hasattr(self.asset, "_cached_materials"):
            self.asset._cached_materials = view.get_material_properties()
        materials = self.asset._cached_materials

        # update material buffer with new samples
        if self.num_shapes_per_body is not None:
            for body_id in self.asset_cfg.body_ids:
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]
                materials[env_ids, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
        else:
            materials[env_ids] = material_samples[:]

        # apply to simulation
        view.set_material_properties(materials, env_ids)
