# Figure Descriptions

## `pca_representations.png`
Two-dimensional PCA projection of validation representations. Colors indicate digit classes, showing dominant global variance structure and coarse class alignment.

## `lda_representations.png`
Two-dimensional LDA projection using class labels. This view emphasizes class-discriminative structure and highlights linear separability in the learned latent space.

## `spectral_representations.png`
Laplacian-eigenmap (spectral) projection of latent vectors based on a cosine k-nearest-neighbor graph. This visualization emphasizes local manifold neighborhoods and nonlinear cluster geometry.

## `first_layer_top64_rf.png`
Top-64 first-layer receptive fields selected by weight L2 norm. Robust nonlinear contrast scaling is used to improve visibility of localized and oriented patterns.

## `top_selective_neurons_heatmap.png`
Class-conditional mean activations for the most selective neurons in the representation. Columns are neurons sorted by selectivity; rows are classes.

## `confusion_matrix_normalized.png`
Row-normalized confusion matrix of linear-head predictions on validation data. Diagonal intensity reflects per-class recall.

## `class_prototype_cosine.png`
Cosine similarity matrix of class prototype vectors (class-mean latent embeddings). Off-diagonal values indicate inter-class representational overlap.

## `pairwise_cosine_distribution.png`
Distribution of pairwise cosine similarities for same-class versus different-class sample pairs. The separation margin summarizes latent compactness and class separation.

## `representation_metrics.json`
Scalar summary metrics used for quantitative reporting: head validation accuracy, pairwise cosine margins, prototype overlap, and neuron selectivity statistics.
