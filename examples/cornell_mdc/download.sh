#!/bin/bash

dataset=cornell_movie_dialogs_corpus

# download
curl -O http://www.cs.cornell.edu/~cristian/data/{$dataset}.zip
unzip ${dataset}.zip 

# clean-up
datadir=cornell\ movie-dialogs\ corpus
mv ${datadir}/* data
rm -r "${datadir}"
rm -r __MACOSX
rm -f ${dataset}.zip
