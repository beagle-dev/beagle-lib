#!/bin/bash

function test_all_impls {
    #               program        states  taxa   sites  rates  reps   rsrc  rescaling  precision  sse    ctips  rseed  rfreq    root   derivs  lnl_exp  d1_exp   d2_exp   lscalers  ecount   ecomplex  ievect   smatrix

    echo -n "   testing resource=0  precision=SINGLE sse=NO  " 1>&2;
    ./parse_test.sh "genomictest"  "${1}"  "${2}" "${3}" "${4}" "${5}" "0"   "${10}"    "single"   "no"   "${6}" "${7}" "${11}"  "${8}" "${9}"  "${17}"  "${18}"  "${19}"  "${12}"   "${13}"  "${14}"   "${15}"  "${16}" >> benchmark_results.csv

    echo -n "   testing resource=$R precision=SINGLE sse=NO  " 1>&2;
    ./parse_test.sh "genomictest"  "${1}"  "${2}" "${3}" "${4}" "${5}" "$R"  "${10}"    "single"   "no"   "${6}" "${7}" "${11}"  "${8}" "${9}"  "${17}"  "${18}"  "${19}"  "${12}"   "${13}"  "${14}"   "${15}"  "${16}" >> benchmark_results.csv

    echo -n "   testing resource=0  precision=DOUBLE sse=NO  " 1>&2;
    ./parse_test.sh "genomictest"  "${1}"  "${2}" "${3}" "${4}" "${5}" "0"   "${10}"    "double"   "no"   "${6}" "${7}" "${11}"  "${8}" "${9}"  "${17}"  "${18}"  "${19}"  "${12}"   "${13}"  "${14}"   "${15}"  "${16}" >> benchmark_results.csv

    echo -n "   testing resource=$R precision=DOUBLE sse=NO  " 1>&2;
    ./parse_test.sh "genomictest"  "${1}"  "${2}" "${3}" "${4}" "${5}" "$R"  "${10}"    "double"   "no"   "${6}" "${7}" "${11}"  "${8}" "${9}"  "${17}"  "${18}"  "${19}"  "${12}"   "${13}"  "${14}"   "${15}"  "${16}" >> benchmark_results.csv
}

if [ -z "${1}" ];
then
    R=1
else
    R=${1}
fi

if [ ! -f benchmark_results.csv ]
then
echo "program,states,taxa,sites,rates,reps,rsrc,rescaling,precision,sse,ctips,rseed,rfreq,root,derivs,lscalers,ecount,ecomplex,ievect,smatrix,rsrc_name,impl_name,lnl,lnl_diff,d1,d1_diff,d2,d2_diff,best_run,time_real,time_user,time_sys,cpu,gcc_version,revision,date" >> benchmark_results.csv
fi

set -v


#               states  taxa  sites    rates  reps  ctips  rseed  root   derivs  rescale   rfreq  lscalers  ecount  ecomplex  ievect  smatrix  lnl_exp          d1_exp       d2_exp
test_all_impls  "4"     "20"  "10000"  "4"    "50"  "10"   "1"    "yes"  "no"    "manual"  "25"   "no"      "1"     "no"      "no"    "no"     "-10929.91708"   "0"          "0"

test_all_impls  "64"    "10"  "1000"   "4"    "10"  "5"    "1"    "yes"  "no"    "manual"  "25"   "no"      "1"     "no"      "no"    "no"     "-22293.73532"   "0"          "0"


set +v


######################
