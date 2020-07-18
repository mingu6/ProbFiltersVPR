#!/bin/bash 

# Absolute path to this script
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

cd $SCRIPTPATH
python3 src/data/interpolate_raw_data.py
python3 src/data/create_reference_maps.py
python3 src/data/create_query_traverses.py
