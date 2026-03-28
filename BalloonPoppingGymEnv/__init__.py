from gymnasium.envs.registration import register

register(
    id="BalloonPoppingGymEnv/BalloonPoppingEnv-v0",
    entry_point="BalloonPoppingGymEnv.envs:BalloonPoppingEnv",
)
