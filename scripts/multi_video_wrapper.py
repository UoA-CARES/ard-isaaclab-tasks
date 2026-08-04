import gc
import os
from typing import Any, Callable, Generic, SupportsFloat
import numpy as np

import gymnasium as gym
from gymnasium import error, logger
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.error import DependencyNotInstalled, InvalidProbability



class MultiCameraRecordVideo(
    gym.Wrapper[ObsType, ActType, ObsType, ActType],
    Generic[ObsType, ActType],
    gym.utils.RecordConstructorArgs,
):
    """Modified Gymnasium RecordVideo wrapper that supports multi-camera dictionary feeds.

    Saves synchronized individual MP4s for each camera angle (e.g., top.mp4, cam0.mp4).

    See gymnaisum.wrappers.RecordVideo for original implementation.
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        video_folder: str,
        episode_trigger: Callable[[int], bool] | None = None,
        step_trigger: Callable[[int], bool] | None = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        fps: int | None = None,
        disable_logger: bool = True,
        gc_trigger: Callable[[int], bool] | None = lambda episode: True,
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            step_trigger=step_trigger,
            video_length=video_length,
            name_prefix=name_prefix,
            disable_logger=disable_logger,
        )
        gym.Wrapper.__init__(self, env)

        if env.render_mode in {None, "human", "ansi"}:
            raise ValueError(
                f"Render mode is {env.render_mode}, which is incompatible with RecordVideo.",
                "Initialize your environment with a render_mode that returns an image, such as rgb_array.",
            )

        if episode_trigger is None and step_trigger is None:
            from gymnasium.utils.save_video import capped_cubic_video_schedule

            episode_trigger = capped_cubic_video_schedule

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.disable_logger = disable_logger
        self.gc_trigger = gc_trigger

        self.video_folder = os.path.abspath(video_folder)
        if os.path.isdir(self.video_folder):
            logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder "
                f"(try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        if fps is None:
            fps = self.metadata.get("render_fps", 30)
        self.frames_per_sec: int = fps
        self.name_prefix: str = name_prefix
        self._video_name: str | None = None
        self.video_length: int = video_length if video_length != 0 else float("inf")
        self.recording: bool = False
        
        # Store frames as a dict of lists per camera
        # [camera_name, list of frames]
        self.recorded_frames: dict[str, list[np.ndarray]] = {}

        self.step_id = -1
        self.episode_id = -1

        try:
            import moviepy  # noqa: F401
        except ImportError as e:
            raise error.DependencyNotInstalled(
                'MoviePy is not installed, run `pip install "gymnasium[other]"`'
            ) from e

    def _capture_frame(self):
        assert self.recording, "Cannot capture a frame, recording wasn't started."

        # Call env.render(), which returns a dict {"top": np.ndarray, "cam0": np.ndarray, ...}
        frames_dict = self.env.render()

        if isinstance(frames_dict, dict):
            for cam_name, frame in frames_dict.items():
                if cam_name not in self.recorded_frames:
                    self.recorded_frames[cam_name] = []
                
                if isinstance(frame, np.ndarray):
                    self.recorded_frames[cam_name].append(frame)
                else:
                    logger.warn(
                        f"Expected frame for camera '{cam_name}' to be a numpy array, got {type(frame)}."
                    )
        else:
            self.stop_recording()
            logger.warn(
                f"Recording stopped: expected env.render() to return a dict of frames, got {type(frames_dict)}."
            )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and eventually starts a new recording."""
        obs, info = super().reset(seed=seed, options=options)
        self.episode_id += 1

        if self.recording and self.video_length == float("inf"):
            self.stop_recording()

        if self.episode_trigger and self.episode_trigger(self.episode_id):
            self.start_recording(f"{self.name_prefix}-episode-{self.episode_id}")
        if self.recording:
            self._capture_frame()
            # Check length of the first available camera buffer
            if self._get_max_recorded_length() > self.video_length:
                self.stop_recording()

        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment using action, recording observations if self.recording."""
        obs, rew, terminated, truncated, info = self.env.step(action)
        self.step_id += 1

        if self.step_trigger and self.step_trigger(self.step_id):
            self.start_recording(f"{self.name_prefix}-step-{self.step_id}")
        if self.recording:
            self._capture_frame()

            if self._get_max_recorded_length() > self.video_length:
                self.stop_recording()

        return obs, rew, terminated, truncated, info

    def _get_max_recorded_length(self) -> int:
        """Helper to check current buffer frame count."""
        # Need to get the max frames across all cameras
        if not self.recorded_frames:
            return 0
        return max(len(frames) for frames in self.recorded_frames.values())

    def close(self):
        """Closes the wrapper then the video recorder."""
        super().close()
        if self.recording:
            self.stop_recording()

    def start_recording(self, video_name: str):
        """Start a new recording."""
        if self.recording:
            self.stop_recording()

        self.recording = True
        self._video_name = video_name
        self.recorded_frames = {}

    def stop_recording(self):
        """Stop current recording and save individual MP4 files for each camera angle."""
        assert self.recording, "stop_recording was called, but no recording was started"

        if self._get_max_recorded_length() == 0:
            logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError as e:
                raise error.DependencyNotInstalled(
                    'MoviePy is not installed, run `pip install "gymnasium[other]"`'
                ) from e

            moviepy_logger = None if self.disable_logger else "bar"

            #  Iterate over camera dict and export individual MP4s
            for cam_name, frame_list in self.recorded_frames.items():
                if len(frame_list) == 0:
                    continue

                filename = f"{self._video_name}-{cam_name}.mp4"
                path = os.path.join(self.video_folder, filename)

                clip = ImageSequenceClip(frame_list, fps=self.frames_per_sec)
                clip.write_videofile(path, logger=moviepy_logger)
                del clip

        del self.recorded_frames
        self.recorded_frames = {}
        self.recording = False
        self._video_name = None

        if self.gc_trigger and self.gc_trigger(self.episode_id):
            gc.collect()

    def __del__(self):
        """Warn the user in case last video wasn't saved."""
        if self._get_max_recorded_length() > 0:
            logger.warn("Unable to save last video! Did you call close()?")