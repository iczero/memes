# TODO

- buy a bigger GPU
  - become rich
  - win the lottery
  - i dunno lol
- eliminating ponder last time resulted in a better performing model
  - does this still hold true for the latest variant?
  - should probably run two models, one with ponder and one without, for a long
    time and see which one ends up better off
- (DONE) fix gradient accumulation
  - (DONE) construct multiple TrainHelper instances, one per "sub-batch"
- make training more data-efficient
  - maybe try truncated BPTT again
  - (DONE) segment text with heuristics, run each segment instead of only the
    first bit of each document
- run a smaller dataset
  - (done?) extract "high quality" sources from the pile or some other dataset
  - (DONE) retrain tokenizer on that dataset
  - (DONE) definitely attempt to exclude code or Advanced Biology™ from the set
- figure out why reduced precision (bf16) doesn't converge
  - is it residual gating?
  - it works now? no idea why it didn't work before
    - [EXTREMELY LOUD INCORRECT BUZZER]
- update the diagram
  - how long until the updated one ends up outdated again?
- port to torch_xla or jax for TRC
  - this stupid project is now sponsored by google
  - figure out why it does not converge on torch_xla
