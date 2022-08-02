#!/usr/bin/env bash

# Make Data Directory
DATA_DIR=${PWD}/Raw_Data
mkdir $DATA_DIR

# Download Data
wget https://figshare.com/ndownloader/files/21728850 -O $DATA_DIR/chia_with_scope.zip
unzip $DATA_DIR/chia_with_scope.zip -d $DATA_DIR/chia_with_scope/
rm $DATA_DIR/chia_with_scope.zip

wget https://figshare.com/ndownloader/files/21728853 -O $DATA_DIR/chia_without_scope.zip
unzip $DATA_DIR/chia_without_scope.zip -d $DATA_DIR/chia_without_scope/
rm $DATA_DIR/chia_without_scope.zip

# Download Spacy
python3 -m spacy download en_core_web_sm