from baselines.common.atari_wrappers import EpisodicLifeEnv, NoopResetEnv, MaxAndSkipEnv, FireResetEnv, WarpFrame, FrameStack, ScaledFloatFrame


def wrap_env(env, episode_life=False, fire_reset=True):
    if episode_life:
        env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, 30)
    env = MaxAndSkipEnv(env, 4)
    if fire_reset:
        env = FireResetEnv(env)
    env = WarpFrame(env)
    env = FrameStack(env, 4)
    env = ScaledFloatFrame(env)
    return env
