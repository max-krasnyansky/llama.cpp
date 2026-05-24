#!/bin/sh

cd /workspace && \
    cp docs/backend/snapdragon/CMakeUserPresets.json .               && \
    cmake --preset arm64-android-snapdragon-release -B build-android && \
    cmake --build build-android                                      && \
    cmake --install build-android --prefix pkg-adb/llama.cpp
    

