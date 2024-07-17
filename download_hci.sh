#!/bin/bash

ARCHIVE_URL="http://lightfield-analysis.net/hci_database/"

echo
echo "*********************************"
echo "***          PART 1           ***"
echo "*********************************"
echo

# download script for first part
function download_lf_part1 {
    echo
    echo "*********************************"
    echo "***     "$1 " : "$2
    echo "*********************************"
    echo
    mkdir -p HCI_dataset_old/$2
    cd HCI_dataset_old/$2
    if [ ! -s lf.h5 ]
    then
	wget $ARCHIVE_URL/$1/$2/lf.h5
    fi
    cd ../../
    echo
    echo
}

download_lf_part1 blender buddha
download_lf_part1 blender horses
download_lf_part1 blender papillon
download_lf_part1 blender stillLife


echo
echo "*********************************"
echo "***          PART 2           ***"
echo "*********************************"
echo

# download script for first part
function download_lf_part2 {
    echo
    echo "*********************************"
    echo "***     "$1 " : "$2
    echo "*********************************"
    echo
    mkdir -p HCI_dataset_old/$2
    cd HCI_dataset_old/$2


    # ... labels.h5
    if [ ! -s labels.h5 ]
    then
	wget $ARCHIVE_URL/$1/$2/labels.h5
    fi
    cd ../../
    echo
    echo
}

download_lf_part2 blender buddha
download_lf_part2 blender horses
download_lf_part2 blender papillon
download_lf_part2 blender stillLife
