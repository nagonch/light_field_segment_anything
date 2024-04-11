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

matlab -nosplash -nodesktop -r "run('LFToolbox/LFMatlabPathSetup.m'); load('$filename','LF'); LFDispMousePan(LF);waitfor(gcf);exit"