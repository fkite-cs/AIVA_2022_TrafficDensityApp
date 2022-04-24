#!/bin/bash

host_folder=$1
slash="/"


if [[ "$1" = /* ]]
then
    echo "ruta absoluta"
else
    echo "ruta relativa"
    SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
    host_folder=$SCRIPTPATH$slash$host_folder
fi

container_path="/root/dc/"
out_name="out"
img_path=$container_path$2
out_path=$container_path$out_name

# echo $img_path
# echo $out_path
if [ $# -eq 3 ]
then
    echo "Running app in GPU..."
    # docker run --rm -it -v $1:$container_path maevision_tda:pro $img_path $out_path
    echo "Coming soon..."
else
    echo "Running app in CPU..."
    docker run --rm -it -v $host_folder:$container_path maevision/maevision_tda:v1 $img_path $out_path
fi