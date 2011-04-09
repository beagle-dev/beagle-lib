#!/bin/bash

function test_all_impls {
    #              program        states  taxa  sites  rates  reps  rsrc  rescaling  precision  sse    ctips  rseed  rfreq  root   derivs  lnl_exp  d1_exp   d2_exp

    echo -n "   testing resource=0 precision=SINGLE sse=NO  rescalefreq=0   " 1>&2;
    ./run_test.sh  "genomictest"  "$1"    "$2"  "$3"   "$4"   "$5"  "0"   "none"     "single"   "no"   "$6"   "$7"   "0"    "$8"   "$9"    "${10}"  "${11}"  "${12}"

    echo -n "   testing resource=1 precision=SINGLE sse=NO  rescalefreq=0   " 1>&2;
    ./run_test.sh  "genomictest"  "$1"    "$2"  "$3"   "$4"   "$5"  "1"   "none"     "single"   "no"   "$6"   "$7"   "0"    "$8"   "$9"    "${10}"  "${11}"  "${12}"

    echo -n "   testing resource=0 precision=DOUBLE sse=NO  rescalefreq=0   " 1>&2;
    ./run_test.sh  "genomictest"  "$1"    "$2"  "$3"   "$4"   "$5"  "0"   "none"     "double"   "no"   "$6"   "$7"   "0"    "$8"   "$9"    "${10}"  "${11}"  "${12}"

    echo -n "   testing resource=0 precision=DOUBLE sse=YES rescalefreq=0   " 1>&2;
    ./run_test.sh  "genomictest"  "$1"    "$2"  "$3"   "$4"   "$5"  "0"   "none"     "double"   "yes"  "$6"   "$7"   "0"    "$8"   "$9"    "${10}"  "${11}"  "${12}"

    echo -n "   testing resource=1 precision=DOUBLE sse=NO  rescalefreq=0   " 1>&2;
    ./run_test.sh  "genomictest"  "$1"    "$2"  "$3"   "$4"   "$5"  "1"   "none"     "double"   "no"   "$6"   "$7"   "0"    "$8"   "$9"    "${10}"  "${11}"  "${12}"

    echo -n "   testing resource=0 precision=SINGLE sse=NO  rescalefreq=1   " 1>&2;
    ./run_test.sh  "genomictest"  "$1"    "$2"  "$3"   "$4"   "$5"  "0"   "manual"   "single"   "no"   "$6"   "$7"   "1"    "$8"   "$9"    "${10}"  "${11}"  "${12}"

    echo -n "   testing resource=1 precision=SINGLE sse=NO  rescalefreq=1   " 1>&2;
    ./run_test.sh  "genomictest"  "$1"    "$2"  "$3"   "$4"   "$5"  "1"   "manual"   "single"   "no"   "$6"   "$7"   "1"    "$8"   "$9"    "${10}"  "${11}"  "${12}"

    echo -n "   testing resource=0 precision=DOUBLE sse=NO  rescalefreq=1   " 1>&2;
    ./run_test.sh  "genomictest"  "$1"    "$2"  "$3"   "$4"   "$5"  "0"   "manual"   "double"   "no"   "$6"   "$7"   "1"    "$8"   "$9"    "${10}"  "${11}"  "${12}"

    echo -n "   testing resource=0 precision=DOUBLE sse=YES rescalefreq=1   " 1>&2;
    ./run_test.sh  "genomictest"  "$1"    "$2"  "$3"   "$4"   "$5"  "0"   "manual"   "double"   "yes"  "$6"   "$7"   "1"    "$8"   "$9"    "${10}"  "${11}"  "${12}"

    echo -n "   testing resource=1 precision=DOUBLE sse=NO  rescalefreq=1   " 1>&2;
    ./run_test.sh  "genomictest"  "$1"    "$2"  "$3"   "$4"   "$5"  "1"   "manual"   "double"   "no"   "$6"   "$7"   "1"    "$8"   "$9"    "${10}"  "${11}"  "${12}"

    echo -n "   testing resource=0 precision=SINGLE sse=NO  rescalefreq=100 " 1>&2;
    ./run_test.sh  "genomictest"  "$1"    "$2"  "$3"   "$4"   "$5"  "0"   "manual"   "single"   "no"   "$6"   "$7"   "100"  "$8"   "$9"    "${10}"  "${11}"  "${12}"

    echo -n "   testing resource=1 precision=SINGLE sse=NO  rescalefreq=100 " 1>&2;
    ./run_test.sh  "genomictest"  "$1"    "$2"  "$3"   "$4"   "$5"  "1"   "manual"   "single"   "no"   "$6"   "$7"   "100"  "$8"   "$9"    "${10}"  "${11}"  "${12}"

    echo -n "   testing resource=0 precision=DOUBLE sse=NO  rescalefreq=100 " 1>&2;
    ./run_test.sh  "genomictest"  "$1"    "$2"  "$3"   "$4"   "$5"  "0"   "manual"   "double"   "no"   "$6"   "$7"   "100"  "$8"   "$9"    "${10}"  "${11}"  "${12}"

    echo -n "   testing resource=0 precision=DOUBLE sse=YES rescalefreq=100 " 1>&2;
    ./run_test.sh  "genomictest"  "$1"    "$2"  "$3"   "$4"   "$5"  "0"   "manual"   "double"   "yes"  "$6"   "$7"   "100"  "$8"   "$9"    "${10}"  "${11}"  "${12}"

    echo -n "   testing resource=1 precision=DOUBLE sse=NO  rescalefreq=100 " 1>&2;
    ./run_test.sh  "genomictest"  "$1"    "$2"  "$3"   "$4"   "$5"  "1"   "manual"   "double"   "no"   "$6"   "$7"   "100"  "$8"   "$9"    "${10}"  "${11}"  "${12}"  
}

if [ "$1" == "--help" ]
then
    echo "run_test_batch.sh [--help] [--print-header]"
    exit
elif [ "$1" == "--print-header" ]
then
    echo "program,states,taxa,sites,rates,reps,rsrc,rescaling,precision,sse,ctips,rseed,rfreq,root,derivs,rrsrc_name,impl_name,lnl,lnl_diff,d1,d1_diff,d2,d2_diff,best_run,time_real,time_user,time_sys,0,gcc_version,revision,date"
fi

set -v

#               states  taxa  sites    rates  reps  ctips  rseed  root   derivs  lnl_exp          d1_exp       d2_exp
test_all_impls  "4"     "20"  "10000"  "4"    "5"   "0"    "0"    "yes"  "no"    "-181089.06469"  "0"          "0"

#               states  taxa  sites    rates  reps  ctips  rseed  root   derivs  lnl_exp          d1_exp       d2_exp
test_all_impls  "5"     "11"  "1009"   "3"    "5"   "5"    "3"    "no"   "no"    "-12649.18470"   "0"          "0"

#               states  taxa  sites    rates  reps  ctips  rseed  root   derivs  lnl_exp          d1_exp       d2_exp
test_all_impls  "20"    "13"  "637"    "2"    "5"   "13"   "42"   "no"   "yes"   "-13472.24200"   "-91.31782"  "29.00467"

#               states  taxa  sites    rates  reps  ctips  rseed  root   derivs  lnl_exp          d1_exp       d2_exp
test_all_impls  "61"    "8"   "854"    "1"    "5"   "0"    "1"    "no"   "yes"   "-16167.59491"   "-0.00669"   "0.06059"

#               states  taxa  sites    rates  reps  ctips  rseed  root   derivs  lnl_exp          d1_exp       d2_exp
test_all_impls  "64"    "10"  "999"    "4"    "5"   "6"    "666"  "yes"  "no"    "-22569.35710"   "0"          "0"

set +v

rm screen_output

#               program        states  taxa  sites    rates  reps   rsrc  rescaling  precision  sse    ctips  rseed  rfreq  root   derivs lnl_exp         d1_exp   d2_exp
#./run_test.sh  "fourtaxon"    "4"     "4"   "1314"   "4"    "500"  "0"   "no"       "single"   "no"   "0"    "0"    "1"    "no"   "no"   "-3970.10422"    "0"      "0"     
