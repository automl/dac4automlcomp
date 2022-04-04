# DAC4AutoML Competition
This is the common python package for the two tracks of the DAC4AutoML competition at AutoML-Conf. [Dynamic Algorithm Configuration (DAC)](https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/20-ECAI-DAC.pdf) is a generalisation of the well-known paradigms of [Algorithm Configuration](https://www.jmlr.org/papers/volume23/21-0888/21-0888.pdf) and [Per-Instance Algorithm Configuration] (https://dl.acm.org/doi/10.5555/1860967.1861114) and involves configuring an algorithm dynamically instead of keeping a static configuration throughout its run. The aim of the individual tracks is to apply DAC to: 1) Supervised Learning pipelines and 2) Reinforcement Learning pipelines and achieve State-of-the-Art results in both. Python 3.9 will be the programming language used.

The instructions below can also be found as part of the instructions for the individual tracks in their individual repos.

## Tracks
DAC4RL track repo: https://github.com/automl/DAC4RL

DAC4RL CodaLab page: https://codalab.lisn.upsaclay.fr/competitions/3727

DAC4SGD track repo: https://github.com/automl/DAC4SGD

DAC4SGD CodaLab page: https://codalab.lisn.upsaclay.fr/competitions/3672

## Installation
```
# If using SSH keys:
git clone git@github.com:automl/dac4automlcomp.git
cd dac4automlcomp
pip install -e .
```

## Sample Submissions
Please refer to the individual repos mentioned above for instructions specific to each track.

The Bash script [`prepare_upload.sh`](prepare_upload.sh) may be used to package a submission directory into a `.zip` file ready for submission.

## Docker Containers
To run your experiments in the same runtime environment as the competition servers they will be evaluated on, we provide a [Docker](https://docs.docker.com/engine/install/) container. Please see [the Docker container definition file](ubuntu_codalab_Dockerfile.txt) to see what packages will be available in the runtime environment.

## Discussion Forum
There will be a discussion forum for each of the two competition tracks where the participants can discuss the tracks and the issues regarding them.
