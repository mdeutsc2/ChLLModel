#!/bin/bash

#SBATCH --job-name=ll_gpu_julia
#SBATCH --account=pgs0213               # account for mdeutsc2
#SBATCH --time=10:00:00                  # timeout for the batch job
#SBATCH --nodes=1                       # requesting number of nodes
#SBATCH --ntasks-per-node=1		        # requesting number of cpus/node
#SBATCH --gpus-per-node=1		        # number of gpus/node
#SBATCH --mail-type=END,FAIL            # send email to submitting user on job completion or failure

# loading modules
#module load julia/1.5.3
module load cuda

# executable section
#time ~/julia-1.6.3/bin/julia --math-mode=fast --check-bounds=no --threads=auto  ll_main.jl
time ~/julia-1.6.3/bin/julia ./src/ll_main.jl -t GPU $1

dirname=$(date +'%-m-%-d-%Y')

if [ -d "./$dirname" ]
then
	echo "$dirname exists"
else
	echo "creating $dirname"
	mkdir ./$dirname
fi

for eachfile in $(ls ./*.{csv,xyzv,h5})
do
	eachfile=${eachfile:2}
	if [ -e "./$dirname/$eachfile" ]
	then
		# if file exists in target directory
		echo "$eachfile already exists, not moving"
	else
		echo "moving $eachfile"
		mv $eachfile ./$dirname
	fi
done

cp $1 ./$dirname # copy parameters to directory
zip "working_model_$dirname.zip" ./src/ll*.jl
mv "working_model_$dirname.zip" ./$dirname