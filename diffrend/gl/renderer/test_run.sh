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
./render_server -s ../../../../scenes/basic.json
