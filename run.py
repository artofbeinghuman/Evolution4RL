from es import ES
import click
import json
import datetime


@click.group()
def cli():
    pass


@cli.command()
@click.argument('exp')
def es(exp):
    timestamp = datetime.datetime.now()
    with open(exp, 'r') as f:
        exp = json.loads(f.read())
    worker = ES(exp, rand_num_table_size=2000000,
                step_size=0.005, verbose=True)
    worker(3)
    path = "save/{}-{}".format(exp["env_short"], timestamp)
    worker.save(path)


if __name__ == '__main__':
    cli()
