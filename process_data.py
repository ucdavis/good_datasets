import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import json
import argparse

import src

#ArgumentParser objecct configuration
parser = argparse.ArgumentParser()

parser.add_argument(
    '-p', '--path',
    help = 'Path to process.py and codex.json',
    required = True,
    )

parser.add_argument(
    '-v', '--verbose',
    help = 'Path to process.py and codex.json',
    action = 'store_true',
    )

parser.add_argument(
    '-pl', '--plants',
    help = 'Dont build plants',
    action = 'store_false',
    )

parser.add_argument(
    '-l', '--lines',
    help = 'Dont build lines',
    action = 'store_false',
    )

parser.add_argument(
    '-pr', '--profiles',
    help = 'Dont build profiles',
    action = 'store_false',
    )


def process(**kwargs):

	src.utilities.cprint('\n' + 'Module process_data' + '\n', kwargs['verbose'])

	'''
	Paths and parameters
	'''

	module_path = kwargs['path'] + 'process.py'
	codex_path = kwargs['path'] + 'codex.json'

	'''
	Loading processing module
	'''

	module = src.inputs.write_data.load_module(module_path)

	'''
	Loading data
	'''
	src.utilities.cprint('\n' + 'Loading Data' + '\n', kwargs['verbose'])

	data = src.inputs.load_data.load(codex_path, verbose = kwargs['verbose'])

	'''
	Processing raw data
	'''
	src.utilities.cprint('\n' + 'Processing Data' + '\n', kwargs['verbose'])

	module = src.inputs.write_data.load_module(module_path)

	installed = module.build_installed_plants(data, verbose = kwargs['verbose'])
	optional = module.build_optional_plants(data, verbose = kwargs['verbose'])
	lines = module.build_lines(data, verbose = kwargs['verbose'])
	profiles = module.build_profiles(data, verbose = kwargs['verbose'])

	'''
	Formatting processed data
	'''
	src.utilities.cprint('\n' + 'Formatting Data' + '\n', kwargs['verbose'])

	module = src.inputs.write_data.load_module(module_path)

	installed_data = module.format_installed_plants(installed, verbose = kwargs['verbose'])
	optional_data = module.format_optional_plants(optional, verbose = kwargs['verbose'])
	lines_data = module.format_lines(lines, verbose = kwargs['verbose'])
	profiles_data = module.format_profiles(profiles, verbose = kwargs['verbose'])

	'''
	Writing JSONs
	'''
	src.utilities.cprint('\n' + 'Writing Data' + '\n', kwargs['verbose'])

	module = src.inputs.write_data.load_module(module_path)

	module.write_plants(installed_data + optional_data, verbose = kwargs['verbose'])
	module.write_lines(lines_data, verbose = kwargs['verbose'])
	module.write_profiles(profiles_data, verbose = kwargs['verbose'])


if __name__ == "__main__":

    kwargs = vars(parser.parse_args(sys.argv[1:]))

    process(**kwargs)