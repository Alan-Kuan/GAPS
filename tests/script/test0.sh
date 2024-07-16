#!/bin/bash
max=4096
min=4
app=test0

rm -rf $log

for (( i=${min}; i<=${max}; i=i*2 ))
do
    ./${app} ${i} >> ${app}-${i}-log
done