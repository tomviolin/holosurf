#!/bin/bash

# Run the hologram preparations and reconstruction
# takes a single argument, which is the data set name.
dataset=this

options=""
c=1
for datadir in data/*; do
	options="$options ${datadir#data/} $c"
	c=$((c+1))
done
echo $options
dialog --menu "Choose a dataset to process" 10 30 8 $options 2> /tmp/choice.txt
if [[ $? != 0 ]]; then
	echo "Cancelled"
	exit 1
fi
dataset=$(cat /tmp/choice.txt)


./src/holo-04-pyholo.py data/$dataset/raw/frame*.png  | \
 dialog --gauge "Preparing holograms..." 10 50 0
dialog --clear

