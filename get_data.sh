#!/usr/bin/env bash

mkdir data
cd data
wget http://opihi.cs.uvic.ca/sound/genres.tar.gz
tar -xzvf genres.tar.gz
rm -rf genres.tar.gz
