# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL-Games AlgoObserver that stops training early once the ARD fitness metric plateaus."""

from __future__ import annotations

import logging

import torch
from rl_games.common.algo_observer import IsaacAlgoObserver

logger = logging.getLogger(__name__)

# Matches the key every task writes into `extras["log"]` (see e.g. cartpole_env.py's
# `_log_fitness`), which the RL-Games env wrapper renames "log" -> "episode" in-flight.
FITNESS_KEY = "fitness_function"


class EarlyStoppingAlgoObserver(IsaacAlgoObserver):
    """Extends :class:`IsaacAlgoObserver` with fitness-plateau early stopping.

    Tracks the best ``fitness_function`` reading seen so far over a sliding window of
    ``patience`` evaluations (training epochs). Every epoch, ``after_print_stats`` is where
    :class:`IsaacAlgoObserver` averages the epoch's ``fitness_function`` readings and writes
    them to tensorboard as ``Episode/fitness_function``; this subclass captures that same
    average and compares it against the best seen so far. When ``patience`` consecutive
    epochs pass without a new best, training is stopped by pulling forward RL-Games' own
    ``max_epochs`` exit check (see the ``epoch_num >= self.max_epochs`` branch in
    ``rl_games.common.a2c_common.A2CBase.train``), so checkpointing and shutdown proceed
    exactly as they would for an ordinary max-epochs stop.

    Only construct this when early stopping is actually wanted -- ``patience`` must be a
    positive integer. Callers that want to disable early stopping should use a plain
    :class:`IsaacAlgoObserver` instead.
    """

    def __init__(self, patience: int):
        if patience <= 0:
            raise ValueError(f"patience must be a positive integer, got {patience}")
        super().__init__()
        self.patience = patience
        self.best_value: float | None = None
        self.best_step: int | None = None
        self.evaluations_since_improvement = 0

    def after_print_stats(self, frame: int, epoch_num: int, total_time: float) -> None:  # noqa: D102
        # `ep_infos` is cleared inside `super().after_print_stats`, so read the current
        # epoch's fitness readings before deferring to it.
        fitness = self._read_fitness()

        super().after_print_stats(frame, epoch_num, total_time)

        if fitness is None:
            return

        if self.best_value is None or fitness > self.best_value:
            self.best_value = fitness
            self.best_step = epoch_num
            self.evaluations_since_improvement = 0
        else:
            self.evaluations_since_improvement += 1

        if self.evaluations_since_improvement >= self.patience:
            message = (
                f"[early-stop] fitness_function has not improved for {self.patience} evaluations "
                f"(best={self.best_value:.4f} @ epoch {self.best_step}, "
                f"current={fitness:.4f} @ epoch {epoch_num}). Stopping training early."
            )
            print(message)
            logger.info(message)
            # Pull the existing max-epochs exit path (a2c_common.A2CBase.train) forward to
            # this epoch instead of reimplementing checkpointing/shutdown here.
            self.algo.max_epochs = epoch_num

    def _read_fitness(self) -> float | None:
        """Mirror the mean-over-``ep_infos`` computation `IsaacAlgoObserver` does for tensorboard."""
        if not self.ep_infos or FITNESS_KEY not in self.ep_infos[0]:
            return None
        values = []
        for ep_info in self.ep_infos:
            value = ep_info[FITNESS_KEY]
            if not isinstance(value, torch.Tensor):
                value = torch.tensor([value], dtype=torch.float32)
            values.append(value.reshape(-1).float())
        return torch.cat(values).mean().item()
