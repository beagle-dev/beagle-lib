#!/bin/bash

if [ -z "${1}" ];
then
    echo "Usage: test_resource.sh resource_number"
    echo
    ../examples/genomictest/genomictest --resourcelist
else
    TEST_RESOURCE=${1}
    echo "Testing resource..."
    ../examples/genomictest/genomictest --resourcelist | grep "Resource $TEST_RESOURCE" -A 1 | grep -o --color=never ': .*'
    ./run_tests.sh $TEST_RESOURCE 2>&1 | grep "\*\*\*[^:]*:" -B 2
    echo "All tests completed."
fi
