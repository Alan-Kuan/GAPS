#!/bin/bash
max=65536
min=16
app=test0

for (( i=${min}; i<=${max}; i=i*2 ))
do
    echo "size transmit:" ${i}
    ./${app} ${i} > ${app}-${i}-log
done