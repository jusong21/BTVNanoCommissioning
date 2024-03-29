#!/bin/bash
#dasgoclient --query="dataset=/*/*UL*NanoAOD*v2*/NANO*" >> out.txt

# check cms cert
while :
do
	voms-proxy-info -exists -valid 8:0
	if [ $? -ne 0 ]; then
		voms-proxy-init --voms cms
	else
		echo "Valid proxy cert"
		break
	fi
done

# Search datasets on CMS DAS with keywords in input file
input=$1
while read line; do 
    echo $line
	DATASET_TYPE=${line:(-4)}
	if [ ${DATASET_TYPE} == "USER" ]; then
		echo "Private Datasets!"
    	dasgoclient --query="dataset=$line instance=prod/phys03" >> mc_private.txt
    	#dasgoclient --query="dataset=$line | grep dataset.nevents" >> tmp.txt
	else
		echo "Global Datasets!"
    	dasgoclient --query="dataset=$line" >> mc_global.txt
	fi

done < $input

