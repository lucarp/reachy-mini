"""
Ball Tracker Environment - Shared Gym Environment for RL Mini-Projects

This environment simulates a ball tracking task where the robot must keep
a moving ball centered in its camera view by controlling head movements.

State Space (6D):
    - ball_x, ball_y: Ball position in normalized camera coordinates [-1, 1]
    - head_pitch, head_yaw: Head joint angles in radians
    - ball_vel_x, ball_vel_y: Ball velocity in camera coordinates

Action Space (2D continuous):
    - delta_pitch, delta_yaw: Change in head angles [-0.1, 0.1] radians

Reward:
    - r = -distance_from_center - 0.1 * action_magnitude
    - Bonus: +1.0 if ball within 0.1 of center
    - Penalty: -10.0 if ball exits frame

Episode:
    - Max 300 steps (30 seconds at 10 Hz)
    - Terminates early if ball exits frame for > 2 seconds
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
import time


class BallTrackerEnv(gym.Env):
    """Gymnasium environment for ball tracking with robot head control."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        ball_speed: float = 0.3,
        max_steps: int = 300,
        use_real_robot: bool = False,
        sim_mode: bool = True,
    ):
        """
        Initialize the Ball Tracker environment.

        Args:
            render_mode: How to render ("human" or "rgb_array")
            ball_speed: Speed of ball movement (0.1 = slow, 1.0 = fast)
            max_steps: Maximum episode length
            use_real_robot: Whether to use real robot (default: False)
            sim_mode: Whether to run in simulation mode
        """
        super().__init__()

        self.render_mode = render_mode
        self.ball_speed = ball_speed
        self.max_steps = max_steps
        self.use_real_robot = use_real_robot
        self.sim_mode = sim_mode

        # State space: [ball_x, ball_y, head_pitch, head_yaw, ball_vel_x, ball_vel_y]
        self.observation_space = spaces.Box(
            low=np.array([-2.0, -2.0, -0.5, -0.8, -1.0, -1.0]),
            high=np.array([2.0, 2.0, 0.5, 0.8, 1.0, 1.0]),
            dtype=np.float32
        )

        # Action space: [delta_pitch, delta_yaw]
        self.action_space = spaces.Box(
            low=np.array([-0.1, -0.1]),
            high=np.array([0.1, 0.1]),
            dtype=np.float32
        )

        # Internal state
        self.ball_pos = np.zeros(2, dtype=np.float32)
        self.ball_vel = np.zeros(2, dtype=np.float32)
        self.head_pos = np.zeros(2, dtype=np.float32)  # [pitch, yaw]
        self.step_count = 0
        self.out_of_frame_count = 0

        # Ball movement pattern
        self.ball_pattern = "sinusoidal"  # "sinusoidal" or "random"
        self.ball_phase = 0.0

        # Robot interface (if using real robot)
        self.robot = None
        if use_real_robot:
            try:
                from reachy_mini import ReachyMini
                self.robot = ReachyMini()
                print("✅ Connected to real Reachy Mini robot")
            except Exception as e:
                print(f"⚠️ Failed to connect to robot: {e}")
                print("   Continuing in pure simulation mode")
                self.use_real_robot = False

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)

        # Reset counters
        self.step_count = 0
        self.out_of_frame_count = 0

        # Reset ball to center with random velocity
        self.ball_pos = np.array([0.0, 0.0], dtype=np.float32)
        angle = self.np_random.uniform(0, 2 * np.pi)
        self.ball_vel = self.ball_speed * np.array([np.cos(angle), np.sin(angle)])

        # Reset head to center
        self.head_pos = np.array([0.0, 0.0], dtype=np.float32)

        # Reset ball pattern
        self.ball_phase = 0.0

        # Move real robot to center if applicable
        if self.robot is not None:
            try:
                self.robot.look_at_world(x=0.5, y=0.0, z=0.0, duration=1.0)
                time.sleep(1.0)
            except Exception as e:
                print(f"⚠️ Failed to move robot: {e}")

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: [delta_pitch, delta_yaw] head movement

        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode ended (ball out of frame)
            truncated: Whether episode was truncated (max steps)
            info: Additional information
        """
        # Apply action to head position (clamp to limits)
        self.head_pos += action
        self.head_pos[0] = np.clip(self.head_pos[0], -0.5, 0.5)  # pitch
        self.head_pos[1] = np.clip(self.head_pos[1], -0.8, 0.8)  # yaw

        # Move real robot if applicable
        if self.robot is not None:
            try:
                # Convert head angles to world coordinates
                # pitch > 0 = look down, yaw > 0 = look right
                x = 0.5
                y = -self.head_pos[1] * 0.3  # yaw to y (inverted)
                z = -self.head_pos[0] * 0.3  # pitch to z (inverted)
                self.robot.look_at_world(x=x, y=y, z=z, duration=0.1)
            except Exception as e:
                print(f"⚠️ Robot movement failed: {e}")

        # Update ball position based on pattern
        if self.ball_pattern == "sinusoidal":
            # Sinusoidal pattern (Lissajous curve)
            self.ball_phase += 0.02
            self.ball_pos[0] = 0.6 * np.sin(2 * self.ball_phase)
            self.ball_pos[1] = 0.6 * np.sin(3 * self.ball_phase + np.pi / 4)

            # Calculate velocity from position change
            if self.step_count > 0:
                # Velocity is derivative of position
                self.ball_vel[0] = 0.6 * 2 * 0.02 * np.cos(2 * self.ball_phase)
                self.ball_vel[1] = 0.6 * 3 * 0.02 * np.cos(3 * self.ball_phase + np.pi / 4)
        else:
            # Random walk pattern
            self.ball_vel += self.np_random.normal(0, 0.02, size=2)
            self.ball_vel = np.clip(self.ball_vel, -0.5, 0.5)
            self.ball_pos += self.ball_vel * 0.1

        # Calculate reward
        reward, info = self._calculate_reward(action)

        # Check termination conditions
        self.step_count += 1

        # Ball out of frame?
        ball_distance = np.linalg.norm(self.ball_pos)
        if ball_distance > 1.5:
            self.out_of_frame_count += 1
        else:
            self.out_of_frame_count = 0

        terminated = self.out_of_frame_count > 20  # Out for 2 seconds
        truncated = self.step_count >= self.max_steps

        observation = self._get_observation()
        info.update(self._get_info())

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        return np.array([
            self.ball_pos[0],
            self.ball_pos[1],
            self.head_pos[0],
            self.head_pos[1],
            self.ball_vel[0],
            self.ball_vel[1]
        ], dtype=np.float32)

    def _calculate_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate reward for current state and action.

        Returns:
            reward: Scalar reward value
            info: Breakdown of reward components
        """
        # Distance from ball to center (after head movement compensates)
        # The goal is to center the ball, so we measure ball position in frame
        # (head movement doesn't change ball position in this simplified model)
        distance = np.linalg.norm(self.ball_pos)

        # Distance penalty (negative reward)
        distance_reward = -distance

        # Action magnitude penalty (encourage smooth movements)
        action_magnitude = np.linalg.norm(action)
        action_penalty = -0.1 * action_magnitude

        # Bonus for keeping ball well-centered
        centering_bonus = 1.0 if distance < 0.1 else 0.0

        # Penalty for ball going out of frame
        out_of_frame_penalty = -10.0 if distance > 1.5 else 0.0

        total_reward = (
            distance_reward +
            action_penalty +
            centering_bonus +
            out_of_frame_penalty
        )

        info = {
            "distance_reward": distance_reward,
            "action_penalty": action_penalty,
            "centering_bonus": centering_bonus,
            "out_of_frame_penalty": out_of_frame_penalty,
            "distance": distance,
        }

        return total_reward, info

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state."""
        return {
            "ball_pos": self.ball_pos.copy(),
            "head_pos": self.head_pos.copy(),
            "ball_vel": self.ball_vel.copy(),
            "step_count": self.step_count,
            "out_of_frame_count": self.out_of_frame_count,
        }

    def render(self):
        """Render the environment (optional, for debugging)."""
        if self.render_mode == "human":
            print(f"Step {self.step_count}: Ball at {self.ball_pos}, Head at {self.head_pos}")
        return None

    def close(self):
        """Clean up resources."""
        if self.robot is not None:
            # Return head to center
            try:
                self.robot.look_at_world(x=0.5, y=0.0, z=0.0, duration=1.0)
            except:
                pass
        super().close()


# Discrete action wrapper for DQN
class DiscreteBallTrackerEnv(BallTrackerEnv):
    """
    Wrapper that converts continuous action space to discrete for DQN.

    9 discrete actions:
        0: stay
        1-8: move in 8 directions (N, NE, E, SE, S, SW, W, NW)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Override action space to discrete
        self.action_space = spaces.Discrete(9)

        # Define action mapping
        step_size = 0.05
        self.action_map = {
            0: np.array([0.0, 0.0]),           # stay
            1: np.array([step_size, 0.0]),     # up
            2: np.array([step_size, step_size]),   # up-right
            3: np.array([0.0, step_size]),     # right
            4: np.array([-step_size, step_size]),  # down-right
            5: np.array([-step_size, 0.0]),    # down
            6: np.array([-step_size, -step_size]), # down-left
            7: np.array([0.0, -step_size]),    # left
            8: np.array([step_size, -step_size]),  # up-left
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Convert discrete action to continuous and call parent step."""
        continuous_action = self.action_map[action]
        return super().step(continuous_action)
