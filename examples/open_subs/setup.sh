#!/bin/bash

#curl -O http://nlp.stanford.edu/data/OpenSubData.tar
#curl -O https://raw.githubusercontent.com/jiweil/Neural-Dialogue-Generation/master/data/movie_25000
dataset=en.tar.gz
url=http://opus.lingfil.uu.se/OpenSubtitles/
curl -o data/$dataset $url/$dataset

tar -xzvf data/$dataset

find data/ -name '*gz' -exec gunzip '{}' \;
