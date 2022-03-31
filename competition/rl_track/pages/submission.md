## Submission
We ask you to submit a solution policy, which dynamically adjusts the hyperparameters of
the algorithm being executed.

### Format
If you did not handcraft your policy, the code for generating the policy should also be part of the
submission.
This solution policy needs to be compatible with the evaluation code provided by
us.
You can take a look at our examples on how to structure your submission. TODO
We further ask you to provide a description of your solution, please find more details
on that in the `submission_template/description.tex`. TODO
Your solution should be submitted as a `.zip`-file.

### Evaluation Procedure
We will perform several evaluation runs on all of our test instances (instances not seen during
training, but drawn from the same distribution as the training instances).
The final ranking of the participants will be decided by their average rank on each instance.
Our metric for deciding the rank on a given instance is the mean performance across seeds.
While the competition is running, the performance ranks for the leaderboard will be measured on a validation set.
For the final ranks of the participants, we will use an unseen test set.

TODO Reproducibility bonus?