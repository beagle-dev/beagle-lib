#!/bin/bash

if [ ! -f test_results.csv ]
then
    echo "program,states,taxa,sites,rates,reps,rsrc,rescaling,precision,sse,rsrc_name,impl_name,lnl,best_run,time_real,time_user,time_sys,cpu,gcc_version,revision,date" > test_results.csv
fi

#               program         states  taxa    sites   rates   reps    rsrc    rescaling   precision   sse
./run_test.sh   "genomictest"   "4"     "20"    "1000"  "4"     "5"     "0"     "none"      "single"    "no"    >> test_results.csv
./run_test.sh   "genomictest"   "4"     "20"    "1000"  "4"     "5"     "0"     "none"      "double"    "no"    >> test_results.csv
./run_test.sh   "genomictest"   "4"     "20"    "1000"  "4"     "5"     "0"     "none"      "double"    "yes"   >> test_results.csv
./run_test.sh   "genomictest"   "64"    "10"    "1000"  "4"     "5"     "0"     "none"      "single"    "no"    >> test_results.csv
./run_test.sh   "genomictest"   "64"    "10"    "1000"  "4"     "5"     "0"     "none"      "double"    "no"    >> test_results.csv




