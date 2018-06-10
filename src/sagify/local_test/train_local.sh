#!/bin/sh

image=deep-learning-addition-img
test_path=$1

docker run -v ${test_path}:/opt/ml --rm ${image} train
