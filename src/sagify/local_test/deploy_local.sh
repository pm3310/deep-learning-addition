#!/bin/sh

image=deep-learning-addition-img
test_path=$1

docker run -it -v ${test_path}:/opt/ml -p 8080:8080 --rm ${image} serve
