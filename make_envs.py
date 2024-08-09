import safety_gymnasium
import envs
import numpy as np
import os


def make_env(args):
    env = safety_gymnasium.make(args.env.name)
    return env
