"""
Deployment script for PPO-trained quadruped (state-based, no vision).

Matches the training script:
- Observation: [joint_pos (12), joint_targets (12), vx, pitch, roll, sensor_state]
- Actions: normalized [-1,1] mapped to joint_limits
"""

import time
import numpy as np
import signal
import math

from stable_baselines3 import PPO
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# ===== CONFIG (copied from training) =====
CONFIG = {
    "base_path": "/Cuboid",       # robot base
    "sensor_path": "/SensingNose",
    "joint_paths": [
        "/FL_hip_yaw", "/FL_hip_pitch", "/FL_knee_pitch",
        "/FR_hip_yaw", "/FR_hip_pitch", "/FR_knee_pitch",
        "/BL_hip_yaw", "/BL_hip_pitch", "/BL_knee_pitch",
        "/BR_hip_yaw", "/BR_hip_pitch", "/BR_knee_pitch",
    ],
    "joint_limits": [(-0.25, 0.25)] * 12,
    "frameskip": 10,
    "max_episode_steps": 800,
    "max_abs_pitch": math.radians(35.0),
    "max_abs_roll": math.radians(35.0),
}
MODEL_PATH = "ppo_quadruped_multi.zip"
CONTROL_HZ = 30
PORT = 23000   # single CoppeliaSim instance
# =========================================


class Deployer:
    def __init__(self, cfg, port=23000):
        self.cfg = cfg

        # Connect to CoppeliaSim
        self.client = RemoteAPIClient('localhost', port)
        self.sim = self.client.getObject('sim')
        self.sim.setStepping(True)

        # Resolve objects
        self.base = self.sim.getObject(cfg["base_path"])
        self.sensor = self.sim.getObject(cfg["sensor_path"])
        self.joints = [self.sim.getObject(p) for p in cfg["joint_paths"]]

        # Limits
        self.joint_limits = np.array(cfg["joint_limits"], dtype=np.float32)
        self.joint_targets = np.array([(lo + hi) * 0.5 for lo, hi in self.joint_limits], dtype=np.float32)

        # Warm-up
        for _ in range(10):
            self.sim.step()

    # ---- helpers ----
    def _get_joint_positions(self):
        return np.array([self.sim.getJointPosition(j) for j in self.joints], dtype=np.float32)

    def _get_base_rpy(self):
        a, b, c = self.sim.getObjectOrientation(self.base, -1)
        return float(a), float(b), float(c)

    def _get_base_vel(self):
        v = self.sim.getObjectVelocity(self.base)
        return np.array(v[0], dtype=np.float32)  # linear

    def _read_sensor(self):
        res = self.sim.readProximitySensor(self.sensor)
        det_state = int(res[0]) if res else 0
        return det_state

    def get_obs(self):
        joint_pos = self._get_joint_positions()
        v_linear = self._get_base_vel()
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

    def apply_action(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        lo = self.joint_limits[:, 0]
        hi = self.joint_limits[:, 1]
        self.joint_targets = ((action + 1.0) * 0.5) * (hi - lo) + lo
        for j, tgt in zip(self.joints, self.joint_targets):
            self.sim.setJointTargetPosition(j, float(tgt))

        # advance sim by frameskip steps
        for _ in range(int(self.cfg["frameskip"])):
            self.sim.step()


def main():
    stop_flag = {"stop": False}
    def _sigint(_sig, _frm): stop_flag["stop"] = True
    signal.signal(signal.SIGINT, _sigint)

    # Load policy
    print(f"Loading model: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)

    env = Deployer(CONFIG, port=PORT)

    if env.sim.getSimulationState() == 0:
        env.sim.startSimulation()
        for _ in range(5):
            env.sim.step()

    dt = 1.0 / CONTROL_HZ
    next_t = time.perf_counter()

    print("Running deployment loop. Press Ctrl+C to stop.")
    try:
        while not stop_flag["stop"]:
            obs = env.get_obs()
            action, _ = model.predict(obs, deterministic=True)
            env.apply_action(action)

            # pacing
            next_t += dt
            sleep_dur = next_t - time.perf_counter()
            if sleep_dur > 0:
                time.sleep(sleep_dur)
            else:
                next_t = time.perf_counter()

    finally:
        print("Stopping simulation...")
        if env.sim.getSimulationState() != 0:
            env.sim.stopSimulation()
        print("Done.")


if __name__ == "__main__":
    import signal
    main()
