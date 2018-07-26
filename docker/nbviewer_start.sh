#!/bin/sh

instance_name=nbviewer
host_notebook_dir=$(realpath `dirname $0`/notebooks)
host_port=10002

# start
mkdir -p $host_notebook_dir
sudo docker run --rm -it -d  --name $instance_name -v $host_notebook_dir:/tmp -p $host_port:8080 jupyter/nbviewer python3 -m nbviewer --localfiles=/tmp --port=8080

ret=$?
if [ $ret -eq 0 ]; then
    echo "Docker '$instance_name' start done."
fi
