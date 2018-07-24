#!/bin/sh

instance_name=jupyter_notebook
host_notebook_dir=$(realpath `dirname $0`/notebooks)
host_port=10001

# start
mkdir -p $host_notebook_dir
sudo docker run --rm -it -d -e PASSWORD='' --name $instance_name -p $host_port:8888 -v $host_notebook_dir:/notebooks tensorflow/tensorflow

ret=$?
if [ $ret -eq 0 ]; then
    echo "Docker '$instance_name' start done."
fi
