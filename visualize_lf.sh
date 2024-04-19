#!/bin/bash

while getopts "f:" opt; do
  case ${opt} in
    f )
      filename=${OPTARG}
      ;;
    \? )
      echo "Usage: $0 -f <filename>"
      exit 1
      ;;
  esac
done

if [ -z "$filename" ]; then
  echo "Filename not provided."
  exit 1
fi

python -c "from plenpy.lightfields import LightField; LF = LightField.from_mat_file('segments.mat', key='LF'); LF.show();"