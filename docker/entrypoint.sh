#!/bin/bash

if [ $# -eq 3 ]
then
    python3 main.py --img_path $1 --out_folder $2 --device $3
else
    python3 main.py --img_path $1 --out_folder $2
fi