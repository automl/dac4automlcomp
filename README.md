# DAC4AutoML Competition
This is the common python package for the 2 tracks of the DAC4AutoML competition at AutoML-Conf. The instructions below can also be found as part of the instructions for the individual tracks in their individual repos.

## Tracks
DAC4RL track: https://github.com/automl-private/DAC4RL

DAC4SGD track:

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
To run your experiments in the same runtime environment as the competition servers they will be evaluated on, we provide [Singularity containers](https://sylabs.io/guides/3.5/user-guide/introduction.html).

Please see the individual repos mentioned above to access the individual singularity containers for each track.
