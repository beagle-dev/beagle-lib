#!/bin/bash

function run_genomictest {
    (time ../examples/genomictest/genomictest $1) &> screen_output
    echo >> screen_log
    echo "./genomictest $1" >> screen_log
}

function run_fourtaxon {
    (cd ../examples/fourtaxon && time ./fourtaxon $1) &> screen_output
}

function grep_print_genomictest {
    RSRC_NAME=`grep "Rsrc" screen_output | cut -f 2 -d ":"`
    RSRC_NAME=`echo $RSRC_NAME`
    IMPL_NAME=`grep "Impl" screen_output | cut -f 2 -d ":"`
    IMPL_NAME=`echo $IMPL_NAME`    
    BEST_RUN=`grep "best" screen_output | cut -f 3 -d " " | grep -o [0-9.]*`

    MAX_DIFF=0.01

    LNL=`grep "logL" screen_output | cut -f 3 -d " "`
    LNL_DIFF=`echo \($LNL\) - \($2\) | bc`
    LNL_ERROR=`echo "$LNL_DIFF > $MAX_DIFF || $LNL_DIFF < -$MAX_DIFF" | bc`
    if (( $LNL_ERROR ))
    then
        echo -n "*** SCORING ISSUE: LNL_EXP = $2 LNL = $LNL DIFF = $LNL_DIFF" 1>&2;
    fi

    if [ "$1" == "yes" ]
    then
        D1=`grep "d1" screen_output | cut -f 6 -d " "`
        D2=`grep "d2" screen_output | cut -f 9 -d " "`
        D1_DIFF=`echo \($D1\) - \($3\) | bc`
        D2_DIFF=`echo \($D2\) - \($4\) | bc`
        D1_ERROR=`echo "$D1_DIFF > $MAX_DIFF || $D1_DIFF < -$MAX_DIFF" | bc`
        D2_ERROR=`echo "$D2_DIFF > $MAX_DIFF || $D2_DIFF < -$MAX_DIFF" | bc`
        if (( $D1_ERROR ))
        then
            echo -n "*** SCORING ISSUE: D1_EXP = $3 D1 = $D1 DIFF = $D1_DIFF" 1>&2;
        fi

        if (( $D2_ERROR ))
        then
            echo -n "*** SCORING ISSUE: D2_EXP = $4 D2 = $D2 DIFF = $D2_DIFF" 1>&2;
        fi
    else
        D1="NA"
        D2="NA"
        D1_DIFF="NA"
        D2_DIFF="NA"
    fi

    set -v
    echo -n ","$RSRC_NAME","$IMPL_NAME","$LNL","$LNL_DIFF","$D1","$D1_DIFF","$D2","$D2_DIFF","$BEST_RUN;
    set +v
}

function grep_print_fourtaxon {
    RSRC_NAME=`grep "Rsrc" screen_output | cut -f 2 -d ":"`
    RSRC_NAME=`echo $RSRC_NAME`
    IMPL_NAME=`grep "Impl" screen_output | cut -f 2 -d ":"`
    IMPL_NAME=`echo $IMPL_NAME`    
    LNL=`tail -n 5 screen_output | head -n 1 | grep -o "\-[0-9.]*"`
    BEST_RUN="NA"

    set -v
    echo -n ","$RSRC_NAME","$IMPL_NAME","$LNL","LNL_DIFF","D1","D1_DIFF","D2","D2_DIFF","$BEST_RUN
    set +v
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
        CMD_FLAGS="--states $2 --taxa $3 --sites $4 --rates $5 --reps $6 --rsrc $7 --compact-tips ${11} --seed ${12} --rescale-frequency ${13} --eigencount ${20}"
        if [ "$8" == "manual" ]
        then
            CMD_FLAGS="$CMD_FLAGS --manualscale"
        fi
        if [ "$9" == "double" ]
        then
            CMD_FLAGS="$CMD_FLAGS --doubleprecision"
        fi
        if [ "${10}" == "yes" ]
        then
            CMD_FLAGS="$CMD_FLAGS --SSE"
        fi
        if [ "${14}" == "no" ]
        then
            CMD_FLAGS="$CMD_FLAGS --unrooted"
        fi
        if [ "${15}" == "yes" ]
        then
            CMD_FLAGS="$CMD_FLAGS --calcderivs"
        fi
        if [ "${19}" == "yes" ]
        then
            CMD_FLAGS="$CMD_FLAGS --logscalers"
        fi
        if [ "${21}" == "yes" ]
        then
            CMD_FLAGS="$CMD_FLAGS --eigencomplex"
        fi
        if [ "${22}" == "yes" ]
        then
            CMD_FLAGS="$CMD_FLAGS --ievectrans"
        fi
        if [ "${23}" == "yes" ]
        then
            CMD_FLAGS="$CMD_FLAGS --setmatrix"
        fi

        run_genomictest "$CMD_FLAGS"
    else
        CMD_FLAGS="--niters $6 --rsrc $7"
        if [ "$8" == "none" ]
        then
            CMD_FLAGS="$CMD_FLAGS --scaling 0"
        else
            CMD_FLAGS="$CMD_FLAGS --scaling 1"
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

    if [ -n "`grep "Failed" screen_output`" ]
    then
        echo "** (`grep "Failed" screen_output`)" 1>&2;
    elif [ -n "`grep "error" screen_output`" ]
    then
        echo "*** ERROR: `grep "error" screen_output`" 1>&2;
    else
        set -v; echo -n $1","$2","$3","$4","$5","$6","$7","$8","$9","${10}","${11}","${12}","${13}","${14}","${15}","${19}","${20}","${21}","${22}","${23}; set +v
        if [ "$1" == "genomictest" ]
        then
            grep_print_genomictest ${15} ${16} ${17} ${18}
        else
            grep_print_fourtaxon
        fi
        grep_print_time
        print_system
        set -v; echo; set +v
        echo 1>&2;
    fi

}

if [ -z "${18}" ];
then
    set -v
    echo "parse_test.sh requires 23 arguments, as follows:"
    echo "parse_test.sh <program> <states> <taxa> <sites> <rates> <reps> <rsrc> <rescaling> <precision> <sse> <compact-tips> <rseed> <rescale-frequency> <rooted> <calc-derivs> <lnl-exp> <d1-exp> <d2-exp> <lscalers> <ecount> <ecomplex> <ievect> <smatrix>"
    echo "(see run_tests.sh for examples)"
    set +v
else


    grep_system

    #               program     states  taxa    sites   rates   reps    rsrc    rescaling   precision   sse     ctips   rseed   rfreq   rooted  derivs  lnl_exp  d1_exp  d2_exp  lscalers  ecount  ecomplex  ievect  smatrix
    run_print_test  $1          $2      $3      $4      $5      $6      $7      $8          $9          ${10}   ${11}   ${12}   ${13}   ${14}   ${15}   ${16}    ${17}   ${18}   ${19}     ${20}   ${21}     ${22}   ${23}

    cat screen_output >> screen_log
    rm screen_output
fi





