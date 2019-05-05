#!/bin/bash

# Set up mounts & pass environment and commandline through
nvidia-docker run \
    -v $HOME/Bayes-Nets:/Bayes-Nets \
    -e CUDA_VISIBLE_DEVICES \
    aaditya \
    $*
