#!/bin/bash

temp_dir=$(mktemp -d)
echo $temp_dir
trap 'rm -rf $temp_dir' EXIT

cd $temp_dir
git clone https://github.com/vaik-info/vaik-TensorRT.git
cd ./vaik-TensorRT/tools/tensorflow-quantization
pip install . --no-dependencies