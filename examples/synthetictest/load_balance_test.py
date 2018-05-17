#!/usr/bin/env python2.7

# Daniel Ayres

import sys
import argparse
import subprocess
import re
from math import log, exp, ceil

sites_list   = None
rsrc_list    = None
rsrc_count   = None
timing_list  = None
precision    = None
taxa         = None
rates        = None
states       = None
total_sites  = None
samples      = None
reps         = None
extra_args   = None
resource_re  = None
timing_re    = None
debug_file   = None
cmd_path     = None
best_time_s  = None
best_time    = None
unique_map   = None

def gen_site_list(max_sites, samples):
    samples_list = []
    for i in range(1, samples):
        samples_list.append(max_sites/samples*i)
    samples_list.append(max_sites)
    samples_list.insert(0, 0)
    return samples_list

def unique_resources(rsrc_list):
    # rsrc_list_re = re.compile('Name : (.*)')
    rsrc_list_re = re.compile('Name : (.*?)(?: \(+.*\)+)?$', re.M)

    synthetictest_out = ""
    synthetictest_cmd = [cmd_path]
    synthetictest_cmd.extend(['--resourcelist'])
    try:
        synthetictest_out = subprocess.check_output(synthetictest_cmd)
        all_rsrcs = rsrc_list_re.findall(synthetictest_out)
        all_rsrcs = [all_rsrcs[i] for i in rsrc_list]
        unique_rsrcs = [all_rsrcs[0]]
        unique_map = [0]
        for rsrc in all_rsrcs[1:]:
            is_unique = True
            for unique_rsrc in unique_rsrcs:
                if rsrc == unique_rsrc:
                    is_unique = False
                    unique_map.append(all_rsrcs.index(unique_rsrc))
            if is_unique:
                unique_rsrcs.append(rsrc)
                unique_map.append(all_rsrcs.index(rsrc))
    except subprocess.CalledProcessError:
        print 'ERROR'

    return unique_map


def perform_timing(rsrc, sites_bin):
    timing = 0.0
    sites=sites_list[sites_bin]
    rsrc_number = rsrc_list[rsrc]
    synthetictest_out = ""
    synthetictest_cmd = [cmd_path]
    synthetictest_cmd.extend(['--states', str(states), '--sites', str(sites)])
    synthetictest_cmd.extend(['--taxa', str(taxa), '--reps', str(reps)])
    synthetictest_cmd.extend(['--rates', str(rates), '--rsrc', str(rsrc_number)])
    if precision == 'double':
        synthetictest_cmd.extend(['--doubleprecision'])
    if (sites > 0):
        out_string = str(sites)
        try:
            synthetictest_out = subprocess.check_output(synthetictest_cmd)
            resource = resource_re.search(synthetictest_out)
            if resource:
                out_string += ', ' + resource.group(1)
            else:
                out_string += ', ' + 'resource error'
            timing_capture = timing_re.search(synthetictest_out)

            if timing_capture:
                timing = float(timing_capture.group(1))
                out_string +=  ', ' + timing_capture.group(1)
            else:
                timing = -1
                out_string += ', ' + 'NA'
        except subprocess.CalledProcessError:
            out_string += 'ERROR'
        print out_string    
        debug_file.write('===============================================================\n')
        debug_file.write(out_string + '\n')
        debug_file.write(' '.join(synthetictest_cmd) + '\n')
        debug_file.write(synthetictest_out)
    return timing

def get_timing(rsrc, sites_bin):
    unique_rsrc = unique_map[rsrc]
    if (timing_list[unique_rsrc][sites_bin] < 0):
        timing_list[unique_rsrc][sites_bin] = perform_timing(unique_rsrc, sites_bin)
    return timing_list[unique_rsrc][sites_bin]

def get_time_bound(rsrc, sites_bin):
    timing = 0.0
    unique_rsrc = unique_map[rsrc]
    if (timing_list[unique_rsrc][sites_bin] >= 0):
        timing = timing_list[unique_rsrc][sites_bin]
    else:
        timing = get_time_bound(rsrc, sites_bin-1)
    return timing

def get_weight(rsrc, previous_weight):
    time   = get_timing(rsrc, previous_weight)
    weight = (time/previous_weight if previous_weight > 0 else 0)
    return weight

def get_site_weights(timings):
    weights = [(1.0/time if time > 0 else 0) for time in timings]
    site_weights = [int(round(samples * weight/sum(weights))) for weight in weights]
    remainder = samples - sum(site_weights)
    for i in range(rsrc_count):
        if (remainder != 0):
            if (site_weights[i] != 0):
                site_weights[i] += (1 if remainder > 0 else -1)
                remainder -= (1 if remainder > 0 else -1)
        else:
            break
    return site_weights

def exhaustive(rsrc, sites_count, best_balance, bound_one, bound_two):
    # print timing_list
    best_time = 0.0
    if (rsrc < len(rsrc_list)):
        min_bound = min(bound_one[rsrc], bound_two[rsrc])
        max_bound = max(bound_one[rsrc], bound_two[rsrc])
        # print rsrc_list[0]
        # best_time = 0.0
        time_set = False
        best_iteration = 0
        best_iteration_balance = [0]
        # print "initial rsrc " + str(rsrc) + " best_time " + str(best_time) + " best_iteration " + str(best_iteration)
        # print "searching resource " + str(rsrc) + " from " + str(min_bound) + " to " + str(max_bound)
        for i in range(min_bound, max_bound+1):
            iteration_sites = sites_list[i]
            # print "trying iteration " + str(i) + " sites " + str(iteration_sites) + " sites_count " + str(sites_count)
            if (rsrc < (len(rsrc_list)-1)) or (iteration_sites == sites_count):
                if (time_set == False) or ((time_set == True) and (get_time_bound(rsrc, i) < best_time)):
                    if (iteration_sites <= sites_count):
                        iteration_balance = []
                        current_timing = get_timing(rsrc, i)
                        if (current_timing < best_time) or (time_set == False):
                            balance_time = exhaustive(rsrc+1, sites_count-iteration_sites, iteration_balance, bound_one, bound_two)
                            if (balance_time >= 0):
                                iteration_time = max(current_timing, balance_time)
                                if (time_set == False):
                                    time_set = True
                                    best_time = iteration_time
                                    best_iteration = i
                                    best_iteration_balance = iteration_balance
                                else:
                                    if (iteration_time < best_time):
                                        best_time = iteration_time
                                        best_iteration = i
                                        best_iteration_balance = iteration_balance
                                # print str(rsrc_list[rsrc]) + " accepted iteration " + str(i) + " iteration_time " + str(iteration_time)

        if (time_set == True):
            best_balance.extend(best_iteration_balance)
            best_balance.append(best_iteration)
            # print str(rsrc_list[0]) + " best_time " + str(best_time) + " best_iteration " + str(best_iteration)
        else:
            best_time = -1.0
    else:
        # print "empty rsrc_list"
        if (sites_count > 0):
            best_time = -1.0

    return best_time

def load_balance():
    global best_time
    global best_time_s

    best_time_s = -1.0
    best_time   = -1.0
    for i in reversed(range(rsrc_count)):
        time_s = get_timing(i, samples)
        if (best_time_s < 0) or (time_s < best_time_s):
            best_time_s = time_s
            best_time   = time_s
            best_rsrc   = i
        only_rsrc   = True
        for i in range(rsrc_count):
            if (unique_map[i] != unique_map[best_rsrc]) and (get_timing(i, 1) < best_time_s):
                only_rsrc = False
        if (only_rsrc == True):
            break

    weights = []
    for i in range(rsrc_count):
        if (unique_map[i] != unique_map[best_rsrc]):
            if (get_timing(i, 1) >= best_time_s):
                weights.append(0)
            else:
                weights.append(get_weight(i, 1))
        else:
            weights.append(get_weight(i, samples))

    site_weights = get_site_weights(weights)

    previous_weights = []
    previous_weights.append(site_weights)

    num_bounces = 0
    while True:
        new_weights = get_site_weights([get_weight(i, site_weights[i]) for i in range(rsrc_count)])

        # distribute weights
        unique_rsrcs_set = set(unique_map)
        if (len(unique_rsrcs_set) != len(rsrc_list)):
            for rsrc in unique_rsrcs_set:
                indices = [i for i, x in enumerate(unique_map) if x == rsrc]

                weight_sum = 0
                for index in indices:
                    weight_sum += new_weights[index]
                remainder = weight_sum % len(indices)
                for index in indices:
                    new_weights[index] = weight_sum / len(indices)
                    if (remainder > 0):
                        new_weights[index] += 1
                        remainder -= 1

        new_times = [get_timing(i, site_weights[i]) for i in range(rsrc_count)]
        print str(site_weights) + ": " + str(max(new_times)) + " " + str(new_times) + " => " + str(new_weights)
        if (best_time < 0) or (max(new_times) < best_time):
            best_time = max(new_times)

        seen_before = False
        for previous in reversed(previous_weights):
            if (new_weights == previous):
                if(new_weights != previous_weights[-1]):
                    print "== exhaustive search (looping) =="
                    site_weights = []
                    best_time = exhaustive(0, total_sites, site_weights, new_weights, previous_weights[-1])
                    site_weights.reverse()
                seen_before = True
                break

        if (seen_before == False) and (len(previous_weights) > 1):
            bounce_test = True
            for i in range(rsrc_count):
                new_change     = new_weights[i]          - previous_weights[-1][i]
                new_direction  = (new_change   / abs(new_change))  if new_change  != 0 else 0
                last_change    = previous_weights[-1][i] - previous_weights[-2][i]
                last_direction = (last_change  / abs(last_change)) if last_change != 0 else 0
                # print "new_direction = " + str(new_direction) + " last_direction = " + str(last_direction)
                if (new_direction * last_direction > 0) or (abs(new_change) <= abs(last_change)):
                    bounce_test = False
            if (bounce_test == True):
                num_bounces += 1
                print "num_bounces = " + str(num_bounces)
            if (num_bounces > 1):
                print "== exhaustive search (bouncing) =="
                site_weights = []
                new_time = exhaustive(0, total_sites, site_weights, new_weights, previous_weights[-1])
                if (best_time < 0) or (new_time < best_time):
                    best_time = new_time
                site_weights.reverse()
                seen_before = True

        # if (len(previous_weights) > 2):
        #     print "== exhaustive search (testing) =="
        #     site_weights = []
        #     new_time = exhaustive(0, total_sites, site_weights, new_weights, previous_weights[-1])
        #     if (best_time < 0) or (new_time < best_time):
        #         best_time = new_time
        #     site_weights.reverse()
        #     seen_before = True            


        if (seen_before == False):
            site_weights = new_weights
            previous_weights.append(site_weights)
        else:
            break

    return site_weights


def main():
    global sites_list 
    global rsrc_list  
    global rsrc_count
    global timing_list
    global precision  
    global taxa       
    global rates      
    global states     
    global total_sites
    global samples    
    global reps
    global resource_re
    global timing_re  
    global debug_file
    global cmd_path
    global unique_map

    parser = argparse.ArgumentParser(description='find load balance weights')
    parser.add_argument('synthetictest_path',
                        help='path to synthetictest')
    parser.add_argument('--taxa'            , type=int, required=True ,
                        help='total number of taxa')
    parser.add_argument('--sites'           , type=int, required=True ,
                        help='total number of sites')
    parser.add_argument('--resources'       , type=int, required=True , nargs='+'      , 
                        help='resources to balance')
    parser.add_argument('--samples'         , type=int, required=False, default=4      ,
                        help='number of samples')
    parser.add_argument('--precision'       , type=str, required=False, default='double',
                        help='fp precision')

    args        = parser.parse_args()
    cmd_path    = args.synthetictest_path
    taxa        = args.taxa
    total_sites = args.sites
    rsrc_list   = args.resources
    samples     = args.samples
    precision   = args.precision
    rates       = 4
    states      = 4
    reps        = 10
    samples     = total_sites if samples > total_sites else samples
    sites_rem   = total_sites % samples
    if (sites_rem != 0):
        sample_size = int(round(float(total_sites) / float(samples)))
        samples     = int(ceil( float(total_sites) / float(sample_size)))
        total_sites = int(samples * sample_size)
    sites_list  = gen_site_list(total_sites, samples)
    rsrc_count  = len(rsrc_list)
    unique_map  = unique_resources(rsrc_list)
    resource_re = re.compile('Rsrc Name : (.*?)( \(+.*\)+)?$', re.M)
    timing_re   = re.compile('best run: (.*)s')
    debug_file  = open('debug.txt', 'w')
    timing_list = []
    for rsrc in range(len(unique_map)):
        timing_list.append([])
        timing_list[rsrc].append(float(0))
        for sites in sites_list:
            timing_list[rsrc].append(float(-1))

    balanced_weights = load_balance()
    # balanced_weights = []
    # final_time = exhaustive(0, total_sites, balanced_weights)
    # balanced_weights.reverse()
    # best_time_s = final_time
    # best_balance_sites = [i * int(round(total_sites/samples)) for i in best_balance]
    final_time = max([get_timing(i, balanced_weights[i]) for i in range(rsrc_count)])
    print str(round(best_time_s/final_time,2)) + "x " + str(balanced_weights) + " " + str(final_time) + " (" + str(best_time) + ")"
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

