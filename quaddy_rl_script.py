"""
PPO training script for a quadruped in CoppeliaSim using the ZeroMQ Remote API.
- Single-file: defines a custom Gymnasium environment and a PPO training loop.
- Connects directly to a running CoppeliaSim instance (you must start CoppeliaSim and load your scene first).
- Uses object *names* (paths) to get handles — no command-line handle passing needed.

IMPORTANT:
1) While training, DISABLE any scene script that calls sim.launchExecutable(...) to avoid controller conflicts.
2) In CoppeliaSim, enable stepping from the remote side: this script does it via sim.setStepping(true) and sim.step().
3) Fill out the CONFIG section below with your object names.

Tested with:
- Python 3.9+
- gymnasium>=0.29
- stable-baselines3>=2.3.0
- coppeliasim_zmqremoteapi_client (ships with CoppeliaSim; ensure it's on PYTHONPATH)

Install (once):
    pip install gymnasium stable-baselines3 numpy

Run:
    python ppo_train.py
"""
from __future__ import annotations
import time
import math
from typing import Tuple, Dict, Any, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# CoppeliaSim ZeroMQ Remote API
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# =====================
# ====== CONFIG =======
# =====================
CONFIG = {
    # Root/base object of your robot (used to measure forward progress and orientation):
    "base_path": "/Cuboid",  # <-- CHANGE to your robot's base path

    # Proximity sensor used for obstacle detection:
    "sensor_path": "/SensingNose",

    # Joint object paths (12 joints: FL/FR/BL/BR × {hip_yaw, hip_pitch, knee_pitch})
    # Change these to match your scene exactly:
    "joint_paths": [
        "/FL_hip_yaw", "/FL_hip_pitch", "/FL_knee_pitch",
        "/FR_hip_yaw", "/FR_hip_pitch", "/FR_knee_pitch",
        "/BL_hip_yaw", "/BL_hip_pitch", "/BL_knee_pitch",
        "/BR_hip_yaw", "/BR_hip_pitch", "/BR_knee_pitch",
    ],

    # Joint limits in radians for each of the 12 joints (lo, hi). If None, uses symmetric defaults.
    # Provide 12 pairs or set to None to apply the same default to all.
    "joint_limits": [(-0.785, 0.785)] * 12,  # e.g., [(-0.7, 0.7)] * 12

    # Action scaling: actions ∈ [-1,1] are mapped to target positions within joint limits.
    "action_clip": 1.0,

    # How many internal sim steps per environment step (frameskip):
    "frameskip": 10,

    # Episode length (env steps):
    "max_episode_steps": 800,

    # Terminate if the robot tips over (rad):
    "max_abs_pitch": math.radians(35.0),
    "max_abs_roll": math.radians(35.0),

    # Reward weights:
    "w_forward": 5.0,     # forward progress
    "w_energy": 0.002,    # penalize large actions
    "w_smooth": 0.001,    # penalize action changes
    "w_collision": 1.0,   # penalty if sensor detects obstacle

    # Target simulation step (optional informational):
    "target_dt": 0.05,  # 20 Hz control
}


class HexapodCoppeliaEnv(gym.Env):
    """Custom Gymnasium environment for a quadruped/hexapod in CoppeliaSim."""
    metadata = {"render.modes": []}

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        # Connect to CoppeliaSim
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')

        # ---- Stepping compatibility layer ----
        # Prefer client.setStepping/client.step if available, else fall back to sim.* API.
        self._client_has_step = hasattr(self.client, "step")
        self._client_has_setStepping = hasattr(self.client, "setStepping")
        if self._client_has_setStepping:
            self._set_stepping = self.client.setStepping
        else:
            self._set_stepping = self.sim.setStepping

        if self._client_has_step:
            self._step_fn = self.client.step
        else:
            self._step_fn = self.sim.step

        # Resolve object handles
        self.base = self.sim.getObject(cfg["base_path"])
        self.sensor = self.sim.getObject(cfg["sensor_path"])
        self.joints = [self.sim.getObject(p) for p in cfg["joint_paths"]]
        assert len(self.joints) == 12, "Expected 12 joint paths"

        # Joint limits
        if cfg["joint_limits"] is None:
            lo, hi = -0.8, 0.8
            self.joint_limits = np.array([[lo, hi] for _ in range(12)], dtype=np.float32)
        else:
            arr = np.asarray(cfg["joint_limits"], dtype=np.float32)
            assert arr.shape == (12, 2)
            self.joint_limits = arr

        # Spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        obs_dim = 12 + 12 + 1 + 2 + 1  # 28
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Internal state
        self.prev_action = np.zeros(12, dtype=np.float32)
        self.joint_targets = np.zeros(12, dtype=np.float32)
        self.default_pose = np.array([(lo + hi) * 0.5 for (lo, hi) in self.joint_limits], dtype=np.float32)
        self.episode_step = 0
        self.last_base_pos = None

        # Ensure sim is running
        self._restart_sim()

    # -------------- Core Env API --------------
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)

        # Quick sanity: make sure sim is still running (if user stopped it)
        st = self.sim.getSimulationState()
        # 0=stopped, 16=paused, >0=running (values may vary by version)
        if st == 0:
            self._restart_sim()
        elif hasattr(self.sim, "resumeSimulation") and st == 16:
            self.sim.resumeSimulation()

        # --- Reset robot base pose ---
        self.sim.setObjectPosition(self.base, -1, [0.0, 0.0, 0.2])
        self.sim.setObjectOrientation(self.base, -1, [0.0, 0.0, 0.0])
        # Zero linear & angular velocity
        self.sim.resetDynamicObject(self.base)

        # --- Reset joints ---
        self.joint_targets[:] = self.default_pose
        for j, tgt in zip(self.joints, self.joint_targets):
            # Force state & target to a known configuration
            self.sim.setJointPosition(j, float(tgt))
            self.sim.setJointTargetPosition(j, float(tgt))

        # --- Reset state ---
        self.prev_action[:] = 0.0
        self.episode_step = 0
        self.last_base_pos = self._get_base_pos()

        # --- Let physics settle ---
        for _ in range(10):
            self._step()

        obs = self._get_obs()
        return obs, {}

        print("Sim state:", self.sim.getSimulationState())
        for _ in range(20):
            self._step()
            print("Base pos:", self._get_base_pos())

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Map [-1,1] → [lo,hi]
        lo = self.joint_limits[:, 0]
        hi = self.joint_limits[:, 1]
        self.joint_targets = ((action + 1.0) * 0.5) * (hi - lo) + lo
        self._apply_joint_targets()

        # Frameskip integration
        for _ in range(int(self.cfg["frameskip"])):
            self._step()

        obs = self._get_obs()
        reward, info = self._compute_reward(action)
        self.episode_step += 1

        terminated = self._fallen()
        truncated = self.episode_step >= self.cfg["max_episode_steps"]
        return obs, reward, terminated, truncated, info

    # -------------- Helpers --------------
    def _restart_sim(self):
        """Ensure simulation is running and in stepping mode (only called at init / if stopped)."""
        # Enable synchronous stepping
        self._set_stepping(True)

        st = self.sim.getSimulationState()
        if st == 0:
            # Fresh start
            self.sim.startSimulation()
        elif hasattr(self.sim, "resumeSimulation") and st == 16:
            # If paused, resume
            self.sim.resumeSimulation()
        # Warm up physics
        for _ in range(10):
            self._step()

    def _step(self):
        """Advance one synchronous step, compatible with both API styles."""
        self._step_fn()

    def _apply_joint_targets(self):
        for j, tgt in zip(self.joints, self.joint_targets):
            self.sim.setJointTargetPosition(j, float(tgt))

    def _get_base_pos(self) -> np.ndarray:
        p = self.sim.getObjectPosition(self.base, -1)
        return np.array(p, dtype=np.float32)

    def _get_base_rpy(self) -> Tuple[float, float, float]:
        a, b, c = self.sim.getObjectOrientation(self.base, -1)
        # Depending on your scene, you may want to swap (roll, pitch) order.
        return float(a), float(b), float(c)

    def _get_joint_positions(self) -> np.ndarray:
        pos = [self.sim.getJointPosition(j) for j in self.joints]
        return np.asarray(pos, dtype=np.float32)

    def _read_sensor(self) -> Tuple[int, float]:
        # Different ZMQ bindings return different tuples; handle robustly.
        res = self.sim.readProximitySensor(self.sensor)
        # Accept either (state, dist, pt, handle, normal) or (state, pt, handle, normal)
        det_state = 0
        det_dist = 0.0
        if isinstance(res, (list, tuple)) and len(res) >= 2:
            s = res[0]
            # Some variants put distance at index 1; others put detected point at 1 and no direct distance
            if len(res) >= 5 and isinstance(res[1], (float, int)):
                det_state = int(s)
                det_dist = float(res[1]) if s > 0 else 0.0
            else:
                det_state = int(s)
                det_dist = 0.0
        return det_state, det_dist

    def _get_obs(self) -> np.ndarray:
        joint_pos = self._get_joint_positions()
        v = self.sim.getObjectVelocity(self.base)  # returns (lin, ang)
        v_linear = np.array(v[0], dtype=np.float32)
        vx = v_linear[0]
        roll, pitch, yaw = self._get_base_rpy()
        det_state, det_dist = self._read_sensor()

        obs = np.concatenate([
            joint_pos,
            self.joint_targets.astype(np.float32),
            np.array([vx], dtype=np.float32),
            np.array([pitch, roll], dtype=np.float32),
            np.array([det_dist], dtype=np.float32),
        ], axis=0)
        return obs.astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        info: Dict[str, float] = {}
        # Forward progress along +X
        p = self._get_base_pos()
        if self.last_base_pos is None:
            self.last_base_pos = p.copy()
        # Forward progress along -Y (since robot faces -Y)
        dy = float(self.last_base_pos[1] - p[1])  # positive if moved forward (-Y)
        self.last_base_pos = p

        r_forward = self.cfg["w_forward"] * dy

        r_energy  = -self.cfg["w_energy"] * float(np.sum(np.square(action)))
        r_smooth  = -self.cfg["w_smooth"] * float(np.sum(np.square(action - self.prev_action)))
        self.prev_action = action.copy()
        det_state, _ = self._read_sensor()
        r_collision = -self.cfg["w_collision"] if det_state > 0 else 0.0

        reward = r_forward + r_energy + r_smooth + r_collision
        info.update(dict(r_forward=r_forward, r_energy=r_energy, r_smooth=r_smooth, r_collision=r_collision))
        return reward, info

    def _fallen(self) -> bool:
        roll, pitch, _ = self._get_base_rpy()
        if abs(pitch) > self.cfg["max_abs_pitch"]:
            return True
        if abs(roll) > self.cfg["max_abs_roll"]:
            return True
        return False

    def close(self):
        try:
            if self.sim.getSimulationState() != 0:
                self.sim.stopSimulation()
        except Exception:
            pass



# =====================
# ===== Training ======
# =====================

def make_env() -> gym.Env:
    return HexapodCoppeliaEnv(CONFIG)


def main():
    # Vectorized wrapper (even with 1 env SB3 expects a VecEnv for some algorithms)
    env = DummyVecEnv([make_env])

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
    model.learn(total_timesteps=2048)
    model.save("ppo_hexapod_coppelia.zip")

    env.close()


if __name__ == "__main__":
    main()
