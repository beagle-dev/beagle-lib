#!/usr/bin/env python2.7

# Daniel Ayres

import sys
import argparse
import subprocess
import re
from math import log, exp

def gen_log_site_list(min, max, samples):
    log_range=(log(max) - log(min))
    samples_list = []
    for i in range(0, samples):
        samples_list.append(int(round(exp(log(min) + log_range/(samples-1)*i))))
    return samples_list

def main():
    parser = argparse.ArgumentParser(description='generate synthetictest benchmarks')
    parser.add_argument('synthetictest_path', help='path to synthetictest')
    args = parser.parse_args()

    taxa_list = [16, 400]
    rates = 4
    precision_list = ['double']

    states_list = [4]
    sites_list = gen_log_site_list(100, 100000, 100)
    # sites_list = gen_log_site_list(100, 10000, 4)
    rsrc_list = ['cpu', 'cpu-threaded', 'pll', 'pll-repeats', 'gpu']
    reps = 10

    seed_list = range(1,101)
    # seed_list = range(1,2)

    extra_args = ['--randomtree', '--stdrand']

    timing_re     = re.compile(    'best run: (.*) ms')
    timing_pll_re = re.compile('pll best run: (.*) ms')

    debug_file = open('debug.txt', 'w')

    header = 'iteration, precision, states, sites, taxa, seed, resource, time' 
    print header

    iteration = 0

    for rsrc in rsrc_list:
        for precision in precision_list:
            for states in states_list:
                for taxa in taxa_list:
                    for sites in sites_list:
                        for seed in seed_list:
                            out_string = str(iteration)
                            out_string += ', ' + str(precision)
                            out_string += ', ' + str(states)
                            out_string += ', ' + str(sites)
                            out_string += ', ' + str(taxa)
                            out_string += ', ' + str(seed)
                            synthetictest_cmd = [args.synthetictest_path]
                            synthetictest_cmd.extend(['--states', str(states), '--sites', str(sites)])
                            synthetictest_cmd.extend(['--taxa', str(taxa), '--compacttips', str(taxa)])
                            synthetictest_cmd.extend(['--reps', str(reps), '--rates', str(rates)])
                            iter_timing_re = timing_re
                            if   rsrc == 'cpu':
                                synthetictest_cmd.extend(['--rsrc', '0', '--disablethreads'])
                            elif rsrc == 'cpu-threaded':
                                synthetictest_cmd.extend(['--rsrc', '0'])
                            elif rsrc == 'pll':
                                synthetictest_cmd.extend(['--rsrc', '0', '--plltest'])
                                iter_timing_re = timing_pll_re
                            elif rsrc == 'pll-repeats':
                                synthetictest_cmd.extend(['--rsrc', '0', '--plltest', '--pllrepeats'])
                                iter_timing_re = timing_pll_re
                            elif rsrc == 'gpu':
                                synthetictest_cmd.extend(['--rsrc', '1'])
                            synthetictest_cmd.extend(extra_args)
                            if precision == 'double':
                                synthetictest_cmd.extend(['--doubleprecision'])
                            try:
                                synthetictest_out = subprocess.check_output(synthetictest_cmd)
                                out_string += ', ' + rsrc
                                timing = iter_timing_re.search(synthetictest_out)
                                if timing:
                                    out_string +=  ', ' + timing.group(1)
                                else:
                                    out_string += ', ' + 'NA'
                            except subprocess.CalledProcessError:
                                out_string += 'ERROR'
                            print out_string
                            debug_file.write('===============================================================\n')
                            debug_file.write(out_string + '\n')
                            debug_file.write(' '.join(synthetictest_cmd) + '\n')
                            debug_file.write(synthetictest_out)
                            iteration += 1
    return 0

if __name__ == '__main__':
    sys.exit(main())

