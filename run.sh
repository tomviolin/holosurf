!/bin/bash

# Run the hologram preparations and reconstruction
# takes a single argument, which is the data set name.
dataset=this
if [ "$1" != "" ]; then
	dataset=$1
fi

rm -f data/$dataset/cimage/* data/$dataset/figs/*
./src/holo-04-pyholo.py data/$dataset/holo/frame*.png
#for f in data/this/holo/frame*.png; do
	#./src/holo-04-dpdif.py $f
#done

#cat data/this/cimage/*.png | ffmpeg -f png_pipe -i - test.mp4 -y
