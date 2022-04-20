#!/bin/bash


container_path="/root/dc/"
out_name="out"
img_path=$container_path$2
out_path=$container_path$out_name

echo $img_path
echo $out_path

docker run --rm -it -v $1:$container_path maevision_tda:pro $img_path $out_path