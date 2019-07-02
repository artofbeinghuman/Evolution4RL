import gym
from gym_wrappers import wrap_env
import time
import click

envs = ['AlienNoFrameskip-v4', 'AsterixNoFrameskip-v4', 'AsteroidsNoFrameskip-v4', 'AtlantisNoFrameskip-v4', 'BattleZoneNoFrameskip-v4', 'BeamRiderNoFrameskip-v4', 'BerzerkNoFrameskip-v4', 'BowlingNoFrameskip-v4', 'BoxingNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'ChopperCommandNoFrameskip-v4', 'CrazyClimberNoFrameskip-v4', 'DefenderNoFrameskip-v4', 'DemonAttackNoFrameskip-v4', 'ElevatorActionNoFrameskip-v4', 'EnduroNoFrameskip-v4', 'FishingDerbyNoFrameskip-v4', 'FreewayNoFrameskip-v4', 'FrostbiteNoFrameskip-v4', 'GravitarNoFrameskip-v4', 'HeroNoFrameskip-v4', 'IceHockeyNoFrameskip-v4', 'JamesbondNoFrameskip-v4', 'KangarooNoFrameskip-v4', 'KrullNoFrameskip-v4', 'KungFuMasterNoFrameskip-v4', 'MontezumaRevengeNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'NameThisGameNoFrameskip-v4', 'PhoenixNoFrameskip-v4', 'PitfallNoFrameskip-v4', 'PongNoFrameskip-v4', 'PrivateEyeNoFrameskip-v4', 'QbertNoFrameskip-v4', 'RiverraidNoFrameskip-v4', 'RoadRunnerNoFrameskip-v4', 'RobotankNoFrameskip-v4', 'SeaquestNoFrameskip-v4', 'SolarisNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4', 'StarGunnerNoFrameskip-v4', 'TimePilotNoFrameskip-v4', 'UpNDownNoFrameskip-v4', 'VentureNoFrameskip-v4', 'YarsRevengeNoFrameskip-v4', 'ZaxxonNoFrameskip-v4', 'JourneyEscapeNoFrameskip-v4', 'AdventureNoFrameskip-v4', 'AirRaidNoFrameskip-v4', 'AmidarNoFrameskip-v4', 'AssaultNoFrameskip-v4', 'BankHeistNoFrameskip-v4', 'CarnivalNoFrameskip-v4', 'CentipedeNoFrameskip-v4', 'DoubleDunkNoFrameskip-v4', 'GopherNoFrameskip-v4', 'PooyanNoFrameskip-v4', 'SkiingNoFrameskip-v4', 'TennisNoFrameskip-v4', 'TutankhamNoFrameskip-v4', 'VideoPinballNoFrameskip-v4', 'WizardOfWorNoFrameskip-v4']
short_names = [s[:-len("NoFrameskip-v4")] for s in envs]

# after end of life your agent needs to hit the fire button
# to get the game to start playing again. If it doesn't learn
# to do this then the game will not progress and looked paused
# like you say it seems.


@click.command()
@click.option('-g', '--game', default="Breakout")
@click.option('-r', '--random', is_flag=True)
def play(game, random):
    game = game.capitalize()
    i = short_names.index(game)
    env = gym.make(envs[i])
    env = wrap_env(env, fire_reset=env.get_action_meanings()[1] == 'FIRE')
    i = 0
    if random:
        for _ in range(3):
            _ = env.reset()
            while True:
                # time.sleep(0.3)
                _, rew, done, info = env.step(env.action_space.sample())  # env.action_space.sample())
                env.render()
                i += 1
                print(i, rew, done, info)
                if done:
                    print("I QUIT!")
                    break

    else:
        for _ in range(1):
            _ = env.reset()
            while i < 3100:
                # time.sleep(0.3)
                _, _, done, info = env.step(3)  # env.action_space.sample())
                env.render()
                i += 1
                print(i, done, info)
                if done:
                    print("I QUIT!")
                    break

            for j in range(18):
                i += 1
                print(j)
                time.sleep(1)
                if done:
                    print("I QUIT!")
                    break
                _, _, done, info = env.step(j)  # env.action_space.sample())
                env.render()

            time.sleep(5)

    env.close()
    print(env.get_action_meanings())


if __name__ == '__main__':
    play()

    # # print(env.get_action_meanings())
    # i = 0
    # for _ in range(2800):
    #     _, _, done, info = env.step(5)
    #     env.render()
    #     i += 1
    #     print(i, done, info)
    #     if done:
    #         print("I QUIT!")
    #         break

    # while True and not done:
    #     time.sleep(0.03)
    #     _, _, done, info = env.step(5)
    #     env.render()
    #     i += 1
    #     print(i, done, info)
    #     if done:
    #         print("I QUIT!")
    #         break

# Frostbite:
# 0 = NOOP
# 1 = FIRE
# 2 = UP
# 3 = RIGHT
# 4 = LEFT
# 5 = DOWN
# 6 = UPRIGHT
# 7 = UPLEFT
# 8 = DOWNRIGHT
# 9 = DOWNLEFT
# 10 = UPFIRE
# 11 = RIGHTFIRE
# 12 = LEFTFIRE
# 13 = DOWNFIRE
# 14 = UPRIGHTFIRE
# 15 = UPLEFTFIRE
# 16 = DOWNRIGHTFIRE
# 17 = DOWNLEFTFIRE


# if __name__ == '__main__':
#     f()
