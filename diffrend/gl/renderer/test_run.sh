#!/usr/bin/env bash

BUILD_DIR=build

if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
fi
echo $BUILD_DIR
mkdir -p "$BUILD_DIR"
cd $BUILD_DIR
cmake ..
make
mkdir -p tmp_out
./render_server -s ../../../../scenes/basic_multiobj.json -t ../../../../scenes/camera_trajectory.json -o ./tmp_out/

