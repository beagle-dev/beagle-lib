#!/bin/bash

if [ ! -f test_results.csv ]
then
    echo "program,states,taxa,sites,rates,reps,rsrc,rescaling,precision,sse,rsrc_name,impl_name,lnl,best_run,time_real,time_user,time_sys,cpu,gcc_version,revision,date" > test_results.csv
fi

set -v

#              program        states  taxa  sites    rates  reps   rsrc  rescaling  precision  sse
./run_test.sh  "genomictest"  "4"     "20"  "10000"  "4"    "50"   "0"   "none"     "single"   "no"  >> test_results.csv
./run_test.sh  "genomictest"  "4"     "20"  "10000"  "4"    "50"   "1"   "none"     "single"   "no"  >> test_results.csv
./run_test.sh  "genomictest"  "4"     "20"  "10000"  "4"    "50"   "0"   "none"     "double"   "no"  >> test_results.csv
./run_test.sh  "genomictest"  "4"     "20"  "10000"  "4"    "50"   "0"   "none"     "double"   "yes" >> test_results.csv
./run_test.sh  "genomictest"  "4"     "20"  "10000"  "4"    "50"   "1"   "none"     "double"   "no"  >> test_results.csv

./run_test.sh  "genomictest"  "64"    "10"  "1000"   "4"    "10"   "0"   "none"     "single"   "no"  >> test_results.csv
./run_test.sh  "genomictest"  "64"    "10"  "1000"   "4"    "10"   "1"   "none"     "single"   "no"  >> test_results.csv
./run_test.sh  "genomictest"  "64"    "10"  "1000"   "4"    "10"   "0"   "none"     "double"   "no"  >> test_results.csv
./run_test.sh  "genomictest"  "64"    "10"  "1000"   "4"    "10"   "0"   "none"     "double"   "yes"  >> test_results.csv
./run_test.sh  "genomictest"  "64"    "10"  "1000"   "4"    "10"   "1"   "none"     "double"   "no"  >> test_results.csv

./run_test.sh  "fourtaxon"    "4"     "4"   "1314"   "4"    "100"  "0"   "none"     "single"   "no"  >> test_results.csv
./run_test.sh  "fourtaxon"    "4"     "4"   "1314"   "4"    "100"  "1"   "none"     "single"   "no"  >> test_results.csv
./run_test.sh  "fourtaxon"    "4"     "4"   "1314"   "4"    "100"  "0"   "none"     "double"   "no"  >> test_results.csv
./run_test.sh  "fourtaxon"    "4"     "4"   "1314"   "4"    "100"  "0"   "none"     "double"   "yes" >> test_results.csv
./run_test.sh  "fourtaxon"    "4"     "4"   "1314"   "4"    "100"  "1"   "none"     "double"   "no"  >> test_results.csv

set +v

rm screen_output
