#!/bin/bash

function test_all_impls {
    #               program        states  taxa   sites  rates  reps   rsrc  rescaling  precision  sse    ctips  rseed  rfreq    root   derivs  lnl_exp  d1_exp   d2_exp   lscalers  ecount   ecomplex  ievect   smatrix

    echo -n "   testing resource=0 precision=SINGLE sse=NO  " 1>&2;
    ./parse_test.sh "genomictest"  "${1}"  "${2}" "${3}" "${4}" "${5}" "0"   "${10}"    "single"   "no"   "${6}" "${7}" "${11}"  "${8}" "${9}"  "${17}"  "${18}"  "${19}"  "${12}"   "${13}"  "${14}"   "${15}"  "${16}" >> test_results.csv

    echo -n "   testing resource=1 precision=SINGLE sse=NO  " 1>&2;
    ./parse_test.sh "genomictest"  "${1}"  "${2}" "${3}" "${4}" "${5}" "1"   "${10}"    "single"   "no"   "${6}" "${7}" "${11}"  "${8}" "${9}"  "${17}"  "${18}"  "${19}"  "${12}"   "${13}"  "${14}"   "${15}"  "${16}" >> test_results.csv

    echo -n "   testing resource=0 precision=DOUBLE sse=NO  " 1>&2;
    ./parse_test.sh "genomictest"  "${1}"  "${2}" "${3}" "${4}" "${5}" "0"   "${10}"    "double"   "no"   "${6}" "${7}" "${11}"  "${8}" "${9}"  "${17}"  "${18}"  "${19}"  "${12}"   "${13}"  "${14}"   "${15}"  "${16}" >> test_results.csv

    echo -n "   testing resource=0 precision=DOUBLE sse=YES " 1>&2;
    ./parse_test.sh "genomictest"  "${1}"  "${2}" "${3}" "${4}" "${5}" "0"   "${10}"    "double"   "yes"  "${6}" "${7}" "${11}"  "${8}" "${9}"  "${17}"  "${18}"  "${19}"  "${12}"   "${13}"  "${14}"   "${15}"  "${16}" >> test_results.csv

    echo -n "   testing resource=1 precision=DOUBLE sse=NO  " 1>&2;
    ./parse_test.sh "genomictest"  "${1}"  "${2}" "${3}" "${4}" "${5}" "1"   "${10}"    "double"   "no"   "${6}" "${7}" "${11}"  "${8}" "${9}"  "${17}"  "${18}"  "${19}"  "${12}"   "${13}"  "${14}"   "${15}"  "${16}" >> test_results.csv
}

if [ ! -f test_results.csv ]
then
    echo "program,states,taxa,sites,rates,reps,rsrc,rescaling,precision,sse,ctips,rseed,rfreq,root,derivs,lscalers,ecount,ecomplex,ievect,smatrix,rsrc_name,impl_name,lnl,lnl_diff,d1,d1_diff,d2,d2_diff,best_run,time_real,time_user,time_sys,cpu,gcc_version,revision,date" >> test_results.csv
fi

rm screen_output

set -v

#               states  taxa  sites    rates  reps  ctips  rseed  root   derivs  rescale   rfreq  lscalers  ecount  ecomplex  ievect  smatrix  lnl_exp          d1_exp       d2_exp
test_all_impls  "4"     "14"  "1240"   "4"    "2"   "7"    "0"    "yes"  "no"    "manual"  "2"    "no"      "1"     "no"      "no"    "no"     "-12986.50361"   "0"          "0"

test_all_impls  "4"     "14"  "1240"   "4"    "2"   "7"    "0"    "yes"  "no"    "manual"  "1"    "yes"     "1"     "no"      "no"    "no"     "-12986.50361"   "0"          "0"

test_all_impls  "4"     "14"  "1240"   "4"    "2"   "7"    "0"    "yes"  "no"    "none"    "2"    "no"      "1"     "no"      "no"    "no"     "-12986.50361"   "0"          "0"

test_all_impls  "4"     "17"  "695"    "4"    "1"   "7"    "0"    "yes"  "no"    "manual"  "1"    "yes"     "1"     "no"      "no"    "yes"    "-517.14914"     "0"          "0"

test_all_impls  "4"     "7"   "739"    "5"    "2"   "7"    "0"    "yes"  "no"    "manual"  "2"    "no"      "3"     "no"      "yes"   "no"     "-3352.16256"    "0"          "0"

test_all_impls  "4"     "7"   "739"    "5"    "2"   "7"    "0"    "yes"  "no"    "none"    "2"    "no"      "3"     "no"      "yes"   "no"     "-3352.16256"    "0"          "0"

test_all_impls  "4"     "10"  "587"    "4"    "2"   "0"    "0"    "no"   "no"    "manual"  "2"    "no"      "1"     "no"      "no"    "no"     "-5385.20838"    "0"          "0"

test_all_impls  "4"     "19"  "1853"   "4"    "2"   "0"    "0"    "yes"  "no"    "manual"  "2"    "no"      "1"     "yes"     "no"    "no"     "-27974.41711"   "0"          "0"

test_all_impls  "4"     "9"   "900"    "2"    "2"   "9"    "0"    "no"   "no"    "manual"  "2"    "no"      "1"     "no"      "no"    "no"     "-6839.95223"    "0"          "0"

test_all_impls  "5"     "8"   "456"    "2"    "2"   "8"    "0"    "no"   "no"    "manual"  "2"    "no"      "4"     "no"      "yes"   "no"     "-3406.48912"    "0"          "0"

test_all_impls  "5"     "8"   "456"    "2"    "2"   "8"    "0"    "no"   "no"    "none"    "2"    "no"      "4"     "no"      "yes"   "no"     "-3406.48912"    "0"          "0"

test_all_impls  "8"     "13"  "637"    "2"    "2"   "13"   "0"    "yes"  "no"    "manual"  "2"    "no"      "1"     "no"      "no"    "no"     "-11892.05122"   "0"          "0"

test_all_impls  "8"     "13"  "637"    "2"    "2"   "13"   "0"    "yes"  "no"    "none"    "2"    "no"      "1"     "no"      "no"    "no"     "-11892.05122"   "0"          "0"

test_all_impls  "11"    "15"  "854"    "1"    "2"   "1"    "0"    "yes"  "no"    "manual"  "2"    "no"      "2"     "no"      "no"    "no"     "-16714.00237"   "0"          "0"

test_all_impls  "11"    "15"  "854"    "1"    "2"   "1"    "0"    "yes"  "no"    "none"    "2"    "no"      "2"     "no"      "no"    "no"     "-16714.00237"   "0"          "0"

test_all_impls  "13"    "5"   "425"    "1"    "1"   "5"    "0"    "no"   "yes"   "none"    "2"    "no"      "1"     "no"      "no"    "yes"    "509.97037"      "249.95152"  "-68.23365"

test_all_impls  "20"    "11"  "315"    "4"    "2"   "8"    "0"    "no"   "no"    "manual"  "2"    "no"      "2"     "no"      "no"    "no"     "-5459.22591"    "0"          "0"

test_all_impls  "61"    "6"   "664"    "1"    "2"   "6"    "0"    "no"   "yes"   "manual"  "2"    "no"      "1"     "no"      "yes"   "no"     "-9482.24975"    "-104.55821" "0.08005"

test_all_impls  "61"    "6"   "664"    "1"    "2"   "6"    "0"    "no"   "yes"   "none"    "2"    "no"      "1"     "no"      "yes"   "no"     "-9482.24975"    "-104.55821" "0.08005"

test_all_impls  "64"    "12"  "399"    "3"    "2"   "12"   "0"    "no"   "yes"   "manual"  "2"    "no"      "1"     "no"      "no"    "no"     "-10943.69031"   "-14.85300"  "-29.26092"

set +v


######################

# TODO: investigate issue with setmatrix + partial-tips
#test_all_impls  "13"    "5"  "425"    "1"    "1"   "1"    "0"    "no"   "yes"    "none"    "2"    "no"      "1"     "no"      "no"    "yes"     "-16714.00237"   "0"          "0"

# TODO: investigate issues with eigencomplex + unrooted and eigencomplex + ratecount != 4
#test_all_impls  "4"     "10"  "587"    "4"    "2"   "0"    "0"    "no"   "no"    "manual"  "2"    "no"      "1"     "yes"     "no"    "no"     "-9427.94740"    "0"          "0"


#                 program        states  taxa  sites    rates  reps   rsrc  rescaling  precision  sse    ctips  rseed  rfreq  root   derivs lnl_exp         d1_exp   d2_exp
#./parse_test.sh  "fourtaxon"    "4"     "4"   "1314"   "4"    "500"  "0"   "no"       "single"   "no"   "0"    "0"    "1"    "no"   "no"   "-3970.10422"    "0"      "0"     
