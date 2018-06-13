"""A script for turning Nathan's .mat files into csv files
"""
import numpy as np
from argparse import ArgumentParser
from scipy.io import loadmat

parser = ArgumentParser()
parser.add_argument('-v', '--variable', default='input')
parser.add_argument('mat')
parser.add_argument('csv')
args = parser.parse_args()

data = loadmat(args.mat)[args.variable]
np.savetxt(args.csv, data, delimiter=',')
