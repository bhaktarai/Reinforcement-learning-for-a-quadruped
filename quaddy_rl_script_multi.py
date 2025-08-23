"""
PPO training script for a quadruped in CoppeliaSim with multiple parallel instances.
- Uses SubprocVecEnv to run multiple robots in separate CoppeliaSim processes.
- Each process must run the same scene but listen on a different ZMQ port.

Before running:
1. Open 4 instances of CoppeliaSim manually.
2. In each, load your robot scene and start simulation (Play button).
3. Configure 'remoteApiConnections.txt' to use unique ports(does it automatically anyway), e.g.:
       portIndex1_port = 23000
       portIndex2_port = 23001
       portIndex3_port = 23002
       portIndex4_port = 23003
4. Run this script. PPO will train using all 4 robots in parallel.
"""

from __future__ import annotations
import math
from typing import Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# CoppeliaSim ZeroMQ Remote API
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# =====================
# ====== CONFIG =======
# =====================
CONFIG = {
    "base_path": "/Cuboid",  # robot base
    "sensor_path": "/SensingNose",
    "joint_paths": [
        "/FL_hip_yaw", "/FL_hip_pitch", "/FL_knee_pitch",
        "/FR_hip_yaw", "/FR_hip_pitch", "/FR_knee_pitch",
        "/BL_hip_yaw", "/BL_hip_pitch", "/BL_knee_pitch",
        "/BR_hip_yaw", "/BR_hip_pitch", "/BR_knee_pitch",
    ],
    "joint_limits": [(-0.25, 0.25)] * 12,
    "frameskip": 10,
    #"max_episode_steps": 800,
    "max_abs_pitch": math.radians(35.0),
    "max_abs_roll": math.radians(35.0),
    # Reward weight (forward-only!)
    #"w_forward": 5.0,
}


class QuadrupedCoppeliaEnv(gym.Env):
    """Custom Gymnasium environment for one robot instance in CoppeliaSim."""
    metadata = {"render.modes": []}

    def __init__(self, cfg: Dict[str, Any], port: int = 23000):
        super().__init__()
        self.cfg = cfg

        # Connect to CoppeliaSim on a specific port
        self.client = RemoteAPIClient('localhost', port)
        self.sim = self.client.getObject('sim')

        # Enable stepping mode
        self.sim.setStepping(True)

        # Resolve objects
        self.base = self.sim.getObject(cfg["base_path"])
        self.sensor = self.sim.getObject(cfg["sensor_path"])
        self.joints = [self.sim.getObject(p) for p in cfg["joint_paths"]]

        # Joint limits
        arr = np.asarray(cfg["joint_limits"], dtype=np.float32)
        self.joint_limits = arr

        # Spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        obs_dim = 12 + 12 + 1 + 2 + 1  # joint_pos + targets + vx(vy in this robot's case as it is facing y direction)
        # + pitch/roll + sensor
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Internal state
        self.prev_action = np.zeros(12, dtype=np.float32)
        self.joint_targets = np.zeros(12, dtype=np.float32)
        self.default_pose = np.array([(lo + hi) * 0.5 for (lo, hi) in self.joint_limits], dtype=np.float32)
        self.episode_step = 0
        self.last_base_pos = None

        # Warm up physics
        for _ in range(10):
            self.sim.step()

    # ---------------- Core API ----------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset base pose
        self.sim.setObjectPosition(self.base, -1, [0.0, 0.0, 0.2])
        self.sim.setObjectOrientation(self.base, -1, [0.0, 0.0, 0.0])
        self.sim.resetDynamicObject(self.base)

        # Reset joints
        self.joint_targets[:] = self.default_pose
        for j, tgt in zip(self.joints, self.joint_targets):
            self.sim.setJointPosition(j, float(tgt))
            self.sim.setJointTargetPosition(j, float(tgt))

        # Reset internal state
        self.prev_action[:] = 0.0
        self.episode_step = 0
        self.last_base_pos = self._get_base_pos()

        # Settle physics
        for _ in range(10):
            self.sim.step()

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        lo = self.joint_limits[:, 0]
        hi = self.joint_limits[:, 1]
        self.joint_targets = ((action + 1.0) * 0.5) * (hi - lo) + lo
        for j, tgt in zip(self.joints, self.joint_targets):
            self.sim.setJointTargetPosition(j, float(tgt))

        for _ in range(int(self.cfg["frameskip"])):
            self.sim.step()

        obs = self._get_obs()
        reward = self._compute_reward()
        self.episode_step += 1

        terminated = self._fallen()
        truncated = self.episode_step >= self.cfg["max_episode_steps"]
        return obs, reward, terminated, truncated, {}

    # ---------------- Helpers ----------------
    def _get_base_pos(self) -> np.ndarray:
        return np.array(self.sim.getObjectPosition(self.base, -1), dtype=np.float32)

    def _get_base_rpy(self) -> Tuple[float, float, float]:
        a, b, c = self.sim.getObjectOrientation(self.base, -1)
        return float(a), float(b), float(c)

    def _get_joint_positions(self) -> np.ndarray:
        return np.array([self.sim.getJointPosition(j) for j in self.joints], dtype=np.float32)

    def _read_sensor(self):
        res = self.sim.readProximitySensor(self.sensor)
        det_state = int(res[0]) if res else 0
        return det_state

    def _get_obs(self) -> np.ndarray:
        joint_pos = self._get_joint_positions()
        v = self.sim.getObjectVelocity(self.base)  # (lin, ang)
        v_linear = np.array(v[0], dtype=np.float32)
        vx = v_linear[0]
        roll, pitch, yaw = self._get_base_rpy()
        det_state = self._read_sensor()

        obs = np.concatenate([
            joint_pos,
            self.joint_targets.astype(np.float32),
            np.array([vx], dtype=np.float32),
            np.array([pitch, roll], dtype=np.float32),
            np.array([det_state], dtype=np.float32),
        ])
        return obs.astype(np.float32)

    def _compute_reward(self) -> float:
        # Reward forward progress along -Y
        p = self._get_base_pos()
        if self.last_base_pos is None:
            self.last_base_pos = p.copy()
        dy = float(self.last_base_pos[1] - p[1])  # positive when moving forward in -Y
        self.last_base_pos = p
        return self.cfg["w_forward"] * dy

    def _fallen(self) -> bool:
        roll, pitch, _ = self._get_base_rpy()
        return abs(pitch) > self.cfg["max_abs_pitch"] or abs(roll) > self.cfg["max_abs_roll"]

    def close(self):
        try:
            if self.sim.getSimulationState() != 0:
                self.sim.stopSimulation()
        except Exception:
            pass


# =====================
# ===== Training ======
# =====================
def make_env_fn(port: int):
    def _init():
        return QuadrupedCoppeliaEnv(CONFIG, port=port)

    return _init


def main():
    # Define the ports for each CoppeliaSim instance
    ports = [23000, 23001, 23002, 23003]

    # Create parallel environments
    env = SubprocVecEnv([make_env_fn(port) for port in ports])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=1024,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./ppo_logs",
        device="auto",
    )

    # Train
    model.learn(total_timesteps=500_000)
    model.save("ppo_quadruped_multi.zip")

    env.close()


if __name__ == "__main__":
    main()
