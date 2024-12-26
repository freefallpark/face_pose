#!/bin/bash

# Display Fast DDS Gen Version
fastddsgen -version

#Parse Inputs (we expect Search Directory to be input, this is the directory the .idl files are in)
if [ -z "$1" ]; then
  echo "-- Usage: $0 <search_directory>"
  exit 1
fi
SEARCH_DIR="$1"
if [ -d "$SEARCH_DIR/autogen" ]; then
  echo "-- $SEARCH_DIR/autogen already exists, continuing."
else
  if ! mkdir "autogen"; then
    echo "-- mkdir failed  to make $SEARCH_DIR/autogen"
    exit 1
  fi
fi

#Find all .idl files in SEARCH_DIR and run fastddsgen on each
find "$SEARCH_DIR" -name "*.idl" | while read -r idl_file; do
  echo "-- Generating files using $idl_file"

  #Run fastddsgen and Check for failure
  if ! fastddsgen "$idl_file" -replace -d "$SEARCH_DIR/autogen"; then
    echo "-- fastddsgen failed of $idl_file"
    exit 1
  fi
done
exit 0
