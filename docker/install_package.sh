#!/bin/sh

instance_name=jupyter_notebook

# Install some packages into docker with command pip/apt-get.
# Note: need to re-run this scrip after docker re-started.

sudo docker exec -it $instance_name pip install graphviz
# sudo docker exec -it $instance_name pip install xxxx
# sudo docker exec -it $instance_name apt-get install xxx
