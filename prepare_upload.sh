#!/bin/bash
# e.g. bash prepare_upload.sh <submission dir>
# Based on https://github.com/rdturnermtl/bbo_challenge_starter_kit/blob/master/prepare_upload.sh

# -e  Exit immediately if a command exits with a non-zero status.
# -x  Print commands and their arguments as they are executed.
set -ex
# The return value of a pipeline is the status of the last command to exit with a non-zero status, or zero if no command exited with a non-zero status
set -o pipefail

# Input args
CODE_DIR=$1

# Setup vars
NAME="upload_$(basename ${CODE_DIR})_$(date +%s)"
TMP_DIR="upload_$(basename ${CODE_DIR})_temp"
# Eliminate final slash
CODE_DIR="$(dirname ${CODE_DIR})/$(basename ${CODE_DIR})"


# Test that all required files exist (solution.py, description.pdf, metadata):
if [ ! -e "${CODE_DIR}/solution.py" ]; then
   echo "No solution.py in the submission. Please add your solution in a file named solution.py."
   exit 1
fi
if [ ! -e "${CODE_DIR}/description.pdf" ]; then
   echo "No description.pdf in the submission. Please add a description of your solution in a file named description.pdf."
   exit 1
fi
if [ ! -e "${CODE_DIR}/metadata" ]; then
   echo "No metadata in the submission. Please add an empty file named metadata."
   exit 1
fi

# Test that the directory and .zip file do not exist yet to avoid clobber. -e is for checking any kind of file.
if [ -e "${TMP_DIR}" ]; then
   echo "${TMP_DIR} already exists. Please either remove it or provide a new name as argument"
   exit 1
fi
if [ -e "$NAME.zip" ]; then
   echo "$NAME already exists. Please either remove it or provide a new name as argument"
   exit 1
fi

# Copy in provided files
cp --recursive --no-clobber ${CODE_DIR} ${TMP_DIR}

# Build the .zip with correct directory structure
(cd ${TMP_DIR} && zip -r ../${NAME}.zip ./*)

# Delete created directory above
rm -rf ${TMP_DIR}

# Display final output for user at end:
# Using + rather than - causes these flags to be turned off.
# The flags can also be used upon invocation of the shell.
# The current set of flags may be found in $-.
set +x

echo "----------------------------------------------------------------"
echo "Built achive for upload"

# List files in the built .zip
unzip -l ./$NAME.zip

echo "For scoring, upload $NAME.zip on the CodaLab competition website."
# echo "https://.com/submissions"
