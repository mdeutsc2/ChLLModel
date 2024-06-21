#!/bin/bash
set -e

dirname=$(date +'%m-%d-%Y') # creates var todays date
orig_pwd=$(pwd)

if [ -d "/mnt/SCRATCH/ALLPS_run/"]
then
	echo "/mnt/SCRATCH/ALLPS_run/ exists"
else
	mkdir /mnt/SCRATCH/ALLPS_run/
fi

if [ -d "/mnt/SCRATCH/ALLPS_run/src/"]
then
	echo "/mnt/SCRATCH/ALLPS_run/src/ exists"
else
	mkdir /mnt/SCRATCH/ALLPS_run/src/
fi

cp -r ./src/ll*.jl /mnt/SCRATCH/ALLPS_run/src
cp $3 /mnt/SCRATCH/ALLPS_run/
cd /mnt/SCRATCH/ALLPS_run/

time julia ./src/ll_main.jl $1 $2 $3
# first arg is the flag for the target
# second arg is the target {CPU|GPU}
# third is the parameter file

# clean up of files
cp $3 ./$dirname # copy parameters to directory
zip "working_model_$dirname.zip" ./src/ll*.jl
mv "working_model_$dirname.zip" ./$dirname
echo "$(pwd)"
