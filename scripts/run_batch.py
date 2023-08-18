import concurrent.futures
from subprocess import Popen
from argparse import ArgumentParser

import time


def run(cmd):
    p = Popen(cmd, shell=True)
    p.wait()


def run_job(cmd):
    print(cmd, flush=True)
    run(cmd)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, default="commands.yaml")
    parser.add_argument('--n-workers', type=int, default=1)
    parser.add_argument('--delay', type=float, default=0)
    args = parser.parse_args()

    if args.delay > 0:
        time.sleep(int(args.delay * 3600))

    with open(args.file) as f:
        content = f.readlines()

    cmds = [x.strip() for x in content]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_workers) as executor:
        for cmd in cmds:
            if not cmd.startswith("#") and not cmd.startswith(" "):
                executor.submit(run_job, cmd)