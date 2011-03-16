#!/bin/bash

function run_genomictest {
    (time ../examples/genomictest/genomictest $1) &> screen_output
}

function run_fourtaxon {
    (cd ../examples/fourtaxon && time ./fourtaxon $1) &> screen_output
}

function grep_print_genomictest {
    RSRC_NAME=`grep "Rsrc" screen_output | cut -f 2 -d ":"`
    RSRC_NAME=`echo $RSRC_NAME`
    IMPL_NAME=`grep "Impl" screen_output | cut -f 2 -d ":"`
    IMPL_NAME=`echo $IMPL_NAME`    
    LNL=`grep "logL" screen_output | cut -f 3 -d " "`
    BEST_RUN=`grep "best" screen_output | cut -f 3 -d " " | grep -o [0-9.]*`

    set -v; echo -n ","$RSRC_NAME","$IMPL_NAME","$LNL","$BEST_RUN; set +v
}

function grep_print_fourtaxon {
    RSRC_NAME=`grep "Rsrc" screen_output | cut -f 2 -d ":"`
    RSRC_NAME=`echo $RSRC_NAME`
    IMPL_NAME=`grep "Impl" screen_output | cut -f 2 -d ":"`
    IMPL_NAME=`echo $IMPL_NAME`    
    LNL=`tail -n 5 screen_output | head -n 1 | grep -o "\-[0-9.]*"`
    BEST_RUN="NA"

    set -v; echo -n ","$RSRC_NAME","$IMPL_NAME","$LNL","$BEST_RUN; set +v
}

function grep_print_time {
    TIME_REAL_MIN=`grep "real" screen_output | cut -f 1 -d "m" | grep -o [0-9.].*`
    TIME_REAL_SEC=`grep "real" screen_output | cut -f 2 -d "m" | grep -o [0-9.]*`
    TIME_REAL=`echo $TIME_REAL_MIN*60 + $TIME_REAL_SEC | bc`
    TIME_REAL=`printf "%f" $TIME_REAL`

    TIME_USER_MIN=`grep "user" screen_output | cut -f 1 -d "m" | grep -o [0-9.].*`
    TIME_USER_SEC=`grep "user" screen_output | cut -f 2 -d "m" | grep -o [0-9.]*`
    TIME_USER=`echo $TIME_USER_MIN*60 + $TIME_USER_SEC | bc`
    TIME_USER=`printf "%f" $TIME_USER`

    TIME_SYS_MIN=`grep "sys" screen_output | cut -f 1 -d "m" | grep -o [0-9.].*`
    TIME_SYS_SEC=`grep "sys" screen_output | cut -f 2 -d "m" | grep -o [0-9.]*`
    TIME_SYS=`echo $TIME_SYS_MIN*60 + $TIME_SYS_SEC | bc`
    TIME_SYS=`printf "%f" $TIME_SYS`

    set -v; echo -n ","$TIME_REAL","$TIME_USER","$TIME_SYS; set +v
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
    else
        CMD_FLAGS="--niters $6 --rsrc $7"
        if [ "$8" == "none" ]
        then
            CMD_FLAGS="$CMD_FLAGS --scaling 0"
        else
            CMD_FLAGS="$CMD_FLAGS --scaling $8"
        fi
        if [ "$9" == "double" ]
        then
            CMD_FLAGS="$CMD_FLAGS --double"
        else
            CMD_FLAGS="$CMD_FLAGS --single"
        fi
        if [ "${10}" == "yes" ]
        then
            CMD_FLAGS="$CMD_FLAGS --sse"
        fi

        run_fourtaxon "$CMD_FLAGS"    
    fi

    if [ -n "`grep -o "Failed" screen_output`" ]
    then
        echo Failed: $1", "$2", "$3", "$4", "$5", "$6", "$7", "$8", "$9", "${10} 1>&2;
    else
        set -v; echo -n $1","$2","$3","$4","$5","$6","$7","$8","$9","${10}; set +v
        if [ "$1" == "genomictest" ]
        then
            grep_print_genomictest
        else
            grep_print_fourtaxon
        fi
        grep_print_time
        print_system
        set -v; echo; set +v
    fi

}

if [ -z "${10}" ];
then
    set -v
    echo "run_test.sh requires 10 arguments, as follows:"
    echo "run_test.sh <program> <states> <taxa> <sites> <rates> <reps> <rsrc> <rescaling> <precision> <sse>"
    echo "(see run_test_batch.sh for examples)"
    set +v
else


    grep_system

    #               program     states  taxa    sites   rates   reps    rsrc    rescaling   precision   sse
    run_print_test  $1          $2      $3      $4      $5      $6      $7      $8          $9          ${10}

fi





