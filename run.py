from es import ES
import click
import json
import datetime
from policy import opt_modes, act_modes

# 62 envs
envs = ['AlienNoFrameskip-v4', 'AsterixNoFrameskip-v4', 'AsteroidsNoFrameskip-v4', 'AtlantisNoFrameskip-v4', 'BattleZoneNoFrameskip-v4', 'BeamRiderNoFrameskip-v4', 'BerzerkNoFrameskip-v4', 'BowlingNoFrameskip-v4', 'BoxingNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'ChopperCommandNoFrameskip-v4', 'CrazyClimberNoFrameskip-v4', 'DefenderNoFrameskip-v4', 'DemonAttackNoFrameskip-v4', 'ElevatorActionNoFrameskip-v4', 'EnduroNoFrameskip-v4', 'FishingDerbyNoFrameskip-v4', 'FreewayNoFrameskip-v4', 'FrostbiteNoFrameskip-v4', 'GravitarNoFrameskip-v4', 'HeroNoFrameskip-v4', 'IceHockeyNoFrameskip-v4', 'JamesbondNoFrameskip-v4', 'KangarooNoFrameskip-v4', 'KrullNoFrameskip-v4', 'KungFuMasterNoFrameskip-v4', 'MontezumaRevengeNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'NameThisGameNoFrameskip-v4', 'PhoenixNoFrameskip-v4', 'PitfallNoFrameskip-v4', 'PongNoFrameskip-v4', 'PrivateEyeNoFrameskip-v4', 'QbertNoFrameskip-v4', 'RiverraidNoFrameskip-v4', 'RoadRunnerNoFrameskip-v4', 'RobotankNoFrameskip-v4', 'SeaquestNoFrameskip-v4', 'SolarisNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4', 'StarGunnerNoFrameskip-v4', 'TimePilotNoFrameskip-v4', 'UpNDownNoFrameskip-v4', 'VentureNoFrameskip-v4', 'YarsRevengeNoFrameskip-v4', 'ZaxxonNoFrameskip-v4', 'JourneyEscapeNoFrameskip-v4', 'AdventureNoFrameskip-v4', 'AirRaidNoFrameskip-v4', 'AmidarNoFrameskip-v4', 'AssaultNoFrameskip-v4', 'BankHeistNoFrameskip-v4', 'CarnivalNoFrameskip-v4', 'CentipedeNoFrameskip-v4', 'DoubleDunkNoFrameskip-v4', 'GopherNoFrameskip-v4', 'PooyanNoFrameskip-v4', 'SkiingNoFrameskip-v4', 'TennisNoFrameskip-v4', 'TutankhamNoFrameskip-v4', 'VideoPinballNoFrameskip-v4', 'WizardOfWorNoFrameskip-v4']
short_names = [s[:-len("NoFrameskip-v4")] for s in envs]


@click.command()
@click.option('-g', '--game', default="Breakout")
@click.option('-r', '--render', is_flag=True)
@click.option('-c', '--config', default="default")
@click.option('-gens', '--generations', default=3)
@click.option('-s', '--sigma', default=0.005)  # noise/perturbation variance
@click.option('--seed', default=123)
@click.option('-rn', '--random_noise_size', default=2000000)
@click.option('-c', '--classic_es', is_flag=True)
@click.option('-a', '--activation', default=-1)
@click.option('--gain', default=1.0)
@click.option('-o', '--optimize', default=-1)
@click.option('-m', '--mutate', default=1)
@click.option('-p', '--num_parents', default=1)
@click.option('-nv', '--no_videos', is_flag=True)
@click.option('-b', '--big_net', is_flag=True)
@click.option('-n', '--novelty', is_flag=True)
def es(game, render, config, generations, sigma, seed, random_noise_size, classic_es,
       activation, gain, optimize, mutate, num_parents, no_videos, big_net, novelty):
    timestamp = datetime.datetime.now()
    optimize = opt_modes[optimize]
    activation = act_modes[activation]

    if config == "default":
        config = "configurations/default_atari_config.json"
        with open(config, 'r') as f:
            config = json.loads(f.read())
        game = game[0].capitalize() + game[1:]
        i = short_names.index(game)
        config['env_id'] = envs[i]
        config['env_short'] = short_names[i]
    else:
        with open(config, 'r') as f:
            config = json.loads(f.read())

    path = "save/{}-{}_{}".format(config["env_short"], str(timestamp.date()), str(timestamp.time()))
    txt = "Log {}.log\n\nWith parameters: \ngame={} ({}) \nconfig={} \ngenerations={} \nsigma={} \nseed={}\nrandom_noise_size={} \
           \nclassic_es={} \n(xavier) gain={} \nactivation={} \noptimize={}\nmutate={} parameters\nnum_parents={} workers\nbig_net={}\nnovelty={}\n\
           ".format(path, config['env_short'], config['env_id'], config, generations, sigma, seed, random_noise_size, classic_es,
                    gain, activation, optimize, "all" if mutate == 1 else "1/{} of".format(mutate),
                    "all" if num_parents == 1 else "1/{} of".format(num_parents), big_net, novelty)

    worker = ES(config, rand_num_table_size=random_noise_size, sigma=sigma, seed=seed, render=render,
                verbose=True, log_path=path, initial_text=txt, classic_es=classic_es, gain=gain,
                activation=activation, optimize=optimize, mutate=mutate, no_videos=no_videos,
                big_net=big_net, novelty=novelty, num_parents=num_parents)
    worker(generations)
    worker.save(path + '.es')


if __name__ == '__main__':
    es()
