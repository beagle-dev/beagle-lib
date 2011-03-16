#!/bin/bash

function run_genomictest {
    (time ../examples/genomictest/genomictest $1) &> screen_output
}

function grep_print_genomictest {
    RSRC_NAME=`grep "Rsrc" screen_output | cut -f 4 -d " "`
    IMPL_NAME=`grep "Impl" screen_output | cut -f 4 -d " "`
    LNL=`grep "logL" screen_output | cut -f 3 -d " "`
    BEST_RUN=`grep "best" screen_output | cut -f 3 -d " "`

    set -v; echo -n ","$RSRC_NAME","$IMPL_NAME","$LNL","$BEST_RUN; set +v
}

function grep_print_time {
    TIMEREAL=`grep "real" screen_output | cut -f 2`
    TIME_USER=`grep "user" screen_output | cut -f 2`
    TIME_SYS=`grep "sys" screen_output | cut -f 2`

    set -v; echo -n ","$TIMEREAL","$TIME_USER","$TIME_SYS; set +v
}

function grep_system {
    if [ -f /proc/cpuinfo ]
    then
        CPU_NAME=`cat /proc/cpuinfo | grep "model name" | head -n 1 | cut -f 2 -d ":"`
        CPU=`echo $CPU_NAME`
    else
        CPU_NAME=`system_profiler SPHardwareDataType | grep "Processor Name" | cut -f 2 -d ":"`
        CPU_FREQ=`system_profiler SPHardwareDataType | grep "Processor Speed" | cut -f 2 -d ":"`
        CPU=`echo $CPU_NAME$CPU_FREQ`
    fi
    GCC_VERSION=`gcc --version | head -n 1`
    REVISION=`(cd .. && svnversion .)`
    DATE=`date "+%Y-%m-%d %H:%M:%S"`
}

function print_system {
    set -v; echo -n ","$CPU","$GCC_VERSION","$REVISION","$DATE; set +v
}

function run_print_test {
    set -v; echo -n $1","$2","$3","$4","$5","$6","$7","$8","$9","${10}; set +v

    if [ "$1" == "genomictest" ]
    then
        CMD_FLAGS="--states $2 --taxa $3 --sites $4 --rates $5 --reps $6 --rsrc $7"
        if [ "$8" != "none" ]
        then
            CMD_FLAGS="$CMD_FLAGS --$8"
        fi
        if [ "$9" == "double" ]
        then
            CMD_FLAGS="$CMD_FLAGS --doubleprecision"
        fi
        if [ "${10}" == "yes" ]
        then
            CMD_FLAGS="$CMD_FLAGS --SSE"
        fi

        run_genomictest "$CMD_FLAGS"
        grep_print_genomictest
        grep_print_time
    fi

    print_system
    
    set -v; echo; set +v
}

if [ -z "${10}" ];
then
    set -v
    echo "run_test.sh requires 10 arguments, as follows:"
    echo "run_test.sh <program> <states> <taxa> <sites> <rates> <reps> <rsrc> <rescaling> <precision> <sse>"
    echo "(see run_genomic_tests.sh or run_fourtaxon_tests.sh for examples)"
    set +v
else


    grep_system

    #               program     states  taxa    sites   rates   reps    rsrc    rescaling   precision   sse
    run_print_test  $1          $2      $3      $4      $5      $6      $7      $8          $9          ${10}

    rm screen_output

fi





