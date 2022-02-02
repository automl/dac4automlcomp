# DAC4AutoML Competition
This is the common python package for the two tracks of the DAC4AutoML competition at AutoML-Conf. [Dynamic Algorithm Configuration (DAC)](#TODO link to paper) is a generalisation of [Algorithm Configuration (AC)](#TODO link to paper) and involves configuring an algorithm dynamically instead of keeping a static configuration throughout its run. The aim of the individual tracks is to apply DAC to: 1) Supervised Learning pipelines and 2) Reinforcement Learning pipelines and achieve State-of-the-Art results in both. Python 3.8 will be the programming language used.

The instructions below can also be found as part of the instructions for the individual tracks in their individual repos.

## Tracks
DAC4RL track: https://github.com/automl-private/DAC4RL

DAC4SGD track: https://github.com/automl-private/DAC4SGD
#TODO Update automl-private mentions

## Installation
```
# If using SSH keys:
git clone git@github.com:automl-private/dac4automlcomp.git
cd dac4automlcomp
pip install -e .
```

## Sample Submissions
Please refer to the individual repos mentioned above for instructions specific to each track.

The Bash script [`prepare_upload.sh`](prepare_upload.sh) may be used to package a submission directory into a `.zip` file ready for submission.

## Singularity Containers
To run your experiments in the same runtime environment as the competition servers they will be evaluated on, we provide [Singularity containers](https://sylabs.io/guides/3.5/user-guide/introduction.html). Please see the `.def` Singularity container definition files in the individual repos to see what additional packages will be available in the runtime environments.

Please see the individual repos mentioned above to access the individual singularity containers for each track.

## Discussion Forum
There will be a discussion forum at #TODO where the participants for the two competition tracks can discuss the tracks and the issues regarding them.
