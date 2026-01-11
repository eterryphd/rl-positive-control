# RL Positive Control - Arithmetic Task

**Lab Notebook:** [Link to Google Doc entry for 2025-12-23]

## Project Goal
Validate RL training pipeline on simple 4-step arithmetic before applying to interleaving task.

## Structure
- `data/` - Training/validation/test datasets
- `scripts/` - Training and evaluation code
- `results/` - Logs, metrics, plots
- `checkpoints/` - Model checkpoints during training


## Summary of checkpoing saving logic
1. Staticly defined save every 20 steps
1. Checkpoints saved if reward exceeds 0.3
1. or represents a new best
1. older ones are pruned
1. After every save event, only the first, secont to last, and most recent valid checkpoints are saved.
1. Checkpoints are of the form that training could be restarted from the checkpoint (there are other versions that's don't save enough information for this)

## Summary of reporting
1. Configuration enagles logging steps at various frequencies. currently set to 1
1. Reward based saving prints report to screen if save performed because of reward increase
1. CheckpiontLeanupCallback reports to screen details about what's being removed


## Things to verify
1. Full model input and output saved for each step in training
1. Validation set should be checked against problems seen in training before doing evaluation
1. Prompt enforces no COT and, when necessary, defines precision. (this shouldn't be necessary with integer x integer)

# GRPO Reinforcement Learning Pipeline Workflow (Non-CoT Focus)

## Todo 
1. How much of this can have validation and test code written 
1. modiy evaluate.py and train.py to share a common answer extraction and preprocessing library. refactor to new utils module.
1. Generate problems dynamically in a way that will survive transition from positive control to interleave
  1. how is validation hanled in this situation?
1. Verify that all prompts sent to and responses received from the models are saved as part of the output
  1. Without doing this we won't have any data to troubleshoot if training doesn't happen.
  1. What are the storage implications of this. surely we can be doing compression on the fly.
1. generate oversized validation set to ensure sufficient size after removing previously observed problems
1. Add timestamp and global setp numbers information to logging
1. script to generate oversized validation dataset.
1. Verify checkpoints are the right kind of checkpoints to resume training if

## Stuff currently being worked on
1. determine how large training data set should be  (currently 2.5k?)
  1. is there a cost to it being 2x this? 10x?
  1. Does training have to be done on the same data for each step? why not different data for each step? Doesnt this inherently remove concerns about "overfitting"? We're trying to train for the operation after all, not the operation on specific numbers.
  1. can this be done in a modular way so we can switch out which approach we're using? The specific approach in the positive control won't the same as interleave, but the general approach of "you haven't seen this one before" can be used. 
1. verify that the current version saves the full prompt given to the model as well as the full output. 
  1. this is very post hoc verification assessment
1. identify the current frequency of saving
  1. I believe this has now been changed such that we're saving only the best of the checkpoints.
  1. There should also be a maximum on this, but I'd like to include a bit of a summary of it
  1. Grok: can you provide a summary of this?
1. identify the current level of reporting.
  1. I think I'm mostly relying on piping currently to save the per run information
1. there was something about changing the learning rate over time. this should now be almost or completely removed
1. how hard is it to integrate with that website that reports training information?
1. review how we assign rewards currently. I think we're approximately providing partial rewards for "you're getting closer"
1. how are we currently extracting anwers. I know for a fact we had issues with precision before (we weren't telling the model how many digits to report). The need for this might be removed now, but we also have a "comma" issue, where the model was reporting commas in some answers, and the answer parser didn't seem to handle this well.
1. training and eval data and generation should be switched over to "three digit numbers multiplied by two digit numbers".  