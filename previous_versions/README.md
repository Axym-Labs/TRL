
### Version History
- V1: Simple vicreg implementation
- V2: No expander
- V3: Applying the loss on all layers separately
- V4: Switching from Conv layers to dense linear layers (compatibility with layer-wise objective)
- V5: Neuron local losses
- V6: Using pairs of the same class as positives
- V7: Applying the similarity loss on neighboring pairs (and therefore chunking the data into same-class chunks)
- V8: Simplifying the variance loss function (such that gradients flow through a simpler computation graph) (main version: v08_var_coarse.py)
- V9: Simplifying the covariance loss equivalently
- V9 (Lateral): Using a pass through a lateral layer (optimized to match the covariance matrix) for the covariance loss
- V9 (Inhibition): Trying to use let the lateral layer predict the actual layer's activations and subtract the predictions (similar to lateral inhibition) but that didn't work out.
- V10: Removing aspects from batch normalization (except the mean normalization)
- V11: Adding an external statistics store to calculate the loss as replacement for per-batch statistics. Tuning parameters for training with a batchsize of 1.
- V12: Adding support for different encoders (in sequence) and recurrent layers
