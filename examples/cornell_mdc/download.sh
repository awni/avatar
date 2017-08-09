#!/bin/bash

dataset=cornell_movie_dialogs_corpus

# download
curl -O http://www.cs.cornell.edu/~cristian/data/{$dataset}.zip
unzip ${dataset}.zip 

# clean-up
mv cornell\ movie-dialogs\ corpus/ data/
rm -rf __MACOSX
rm -f ${dataset}.zip
