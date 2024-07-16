#!/bin/bash
max=4096
min=4
app=test0
log=${app}-log.txt
config_dir=/home/

rm -rf $log

for (( i=${min}; i<=${max}; i=i*2 ))
do
    transmit_size=$i
    ./${app} ${config_dir} ${transmit_size} >> ${log}
done