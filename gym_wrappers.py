from baselines.common.atari_wrappers import EpisodicLifeEnv, NoopResetEnv, MaxAndSkipEnv, FireResetEnv, WarpFrame, FrameStack, ScaledFloatFrame


def wrap_env(env, episode_life=False):
    if episode_life:
        env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, 30)
    env = MaxAndSkipEnv(env, 4)
    if env.unwrapped.get_action_meanings()[1] == 'FIRE':
        env = FireResetEnv(env)
    env = WarpFrame(env)  # , width=84, height=84)
    env = FrameStack(env, 4)
    env = ScaledFloatFrame(env)
    return env
