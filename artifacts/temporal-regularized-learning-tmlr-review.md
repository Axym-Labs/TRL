# Reviewer report: “Temporal Regularized Learning: Self-supervised learning local in space and time”

**Venue format:** Transactions on Machine Learning Research (TMLR)  
**Provisional recommendation:** **Reject**  
**Confidence:** **High (4/5)**  
**Review date:** 2026-08-03

## Review basis

I reviewed the complete 22-page manuscript and appendices at manuscript commit `4137582`, the current code repository, and the experiment commits identified by the submission:

- TeReL and TeReL-S main results: `b39b73fac94e984f089820bec7a421499bcd6c0d`;
- backpropagation main results: `25908634afa795840b0026d8481fa69338e857ec`;
- sequence experiments identified in the repository README: `0e454ac` and `221f22c`.

The report follows the [current TMLR reviewer form and acceptance criteria](https://jmlr.org/tmlr/reviewer-guide.html). Its level of detail and separation of critical from non-critical requests were calibrated against public TMLR/OpenReview reports on adjacent work, particularly [*Blockwise Self-Supervised Learning at Scale*](https://openreview.net/forum?id=M2m618iIPk) and [*A Robust Backpropagation-Free Framework for Images*](https://openreview.net/forum?id=leqr0vQzeN).

## Summary of contributions

The submission proposes Temporal Regularized Learning (TeReL), a layer-local representation-learning procedure motivated as an online, per-neuron instantiation of the variance–invariance–covariance structure of VICReg. Temporal neighbors replace two augmented views in the invariance term; running per-neuron statistics provide a variance-control signal; and a learned same-layer lateral map is intended to approximate a decorrelation signal. The submission also presents TeReL-S, a variant with additional detachment/concurrency choices.

The paper makes three principal claims: (i) this construction yields a spatially and temporally local rule with bounded state; (ii) it connects modern regularized self-supervised learning to Slow Feature Analysis (SFA) and trace learning; and (iii) it obtains competitive MNIST classification results and non-trivial MNIST-Rows next-row prediction without end-to-end backpropagation or backpropagation through time. The empirical evidence consists of MNIST linear-probe results, comparisons to backpropagation and literature-reported Forward-Forward and Equilibrium Propagation numbers, representation visualizations, ablations, and one recurrent comparison.

The high-level question is interesting. A faithful neuron-local adaptation of regularized representation learning could be useful to researchers studying local credit assignment, online learning, and constrained hardware. The manuscript also deserves credit for releasing source code, reporting per-seed MNIST results, and explicitly acknowledging that the experiments are small-scale and use synthetic temporal ordering.

However, the central claims are not presently supported. The exact code commit cited for the main TeReL results contains an optimizer-registration defect that leaves every encoder layer after the first unoptimized. The mathematical “fourfold” gradient does not describe either the stated variance-maintenance loss or the non-detached TeReL implementation. The reported experiments are batched and label-ordered rather than demonstrating the claimed online self-supervised rule. There are also serious comparison and test-protocol problems. These issues affect the paper's main results and require new experiments and a new derivation, not a textual clarification.

## Are the claims made in the submission supported by accurate, convincing, and clear evidence?

**No.**

### Strengths

1. **The research question is worthwhile.** Connecting modern non-contrastive objectives to temporally local learning is potentially useful, especially if the locality and memory properties can be stated precisely and demonstrated in an actual streaming implementation.

2. **The paper exposes useful implementation ingredients.** Detaching population statistics, separating the main and lateral losses, and testing a shifted lateral signal are concrete design choices that can be evaluated independently.

3. **The submission includes more diagnostics than a single accuracy table.** The seed-level MNIST values, receptive-field plots, representation diagnostics, parameter-count table, and negative ablations are useful instincts. The limitations section also correctly acknowledges the narrow dataset scope and optimizer sensitivity.

4. **The broad SFA connection is legitimate.** Penalizing temporal differences while enforcing variance and decorrelation is indeed closely connected to SFA. The paper is directionally right to make that relationship explicit, although it currently overstates what is new about the connection and understates the overlap with prior online SFA work.

### Major concerns

#### 1. The cited main-result code does not train the claimed deep encoders

This is the most serious issue. In the exact commit cited for the main TeReL and TeReL-S results (`b39b73f`), `trl/trainer/encoder.py`, lines 68–74, returns from `configure_optimizers` inside the loop over layers:

```python
all_params = []
for layer in self.encoder.layers:
    layer_params, lat_params = layer.layer_lat_params()
    all_params.extend(layer_params)
    all_params.extend(lat_params)
    return self.optim_cls(all_params, lr=self.lr)
```

Consequently, only the first encoder layer and its first lateral map are registered with the optimizer. In the greedy TeReL setting, gradients are computed for later layers when their turn arrives, but those parameters are never stepped. In TeReL-S, later-layer losses are included, but the corresponding parameters are likewise absent from the optimizer. This bug was only fixed later, in commit `0e454ac`, after the commit identified for the main table.

The consequence is direct: the `512,256`, `3 × 500`, and `4 × 2000` TeReL models in Table 2 cannot be interpreted as trained two-, three-, and four-layer encoders. Yet the linear head concatenates activations from all hidden layers (`trl/trainer/head.py`, lines 15–28), including the untrained layers. The claims that performance improves with model scale, that the backpropagation gap closes to 0.13 percentage points at four layers, and that later layers develop specialized units are therefore not established by the cited runs. The auxiliary-parameter counts also describe maps that, after the first layer, were not optimized.

This problem is especially consequential because the paper reports that an entirely untrained encoder already reaches approximately 90% with the chosen linear-probe setup. A head with access to trained first-layer features plus random downstream features can score well without demonstrating successful deep local learning.

All main feedforward results, representation analyses, and claims about depth must be rerun from an immutable corrected commit. At minimum, the authors should log parameter deltas or checksums per layer and report both last-layer-only and all-layer probes against matched random-feature controls.

#### 2. The main gradient proposition is not the gradient of the implemented TeReL objective

The proposition “Gradient as weighted sum” and its appendix define

\[
L=\lambda_S(z-p)^2+\lambda_V(z-m)^2+\lambda_C z\,\mathrm{lat}(\operatorname{detach}(z-m)),
\]

which yields a positive term (+2\lambda_V(z-m)). With positive \(\lambda_V\), this term *reduces* variance and favors collapse; it is not a variance-maintenance term.

The cited implementation instead uses, up to reductions,

\[
L_V=-g(v)(z-m)^2,
\qquad
g(v)=\operatorname{ReLU}(\tau-v),
\]

where the stored variance and gate are detached. Its contribution is therefore

\[
-2\lambda_V g(v)(z-m),
\]

with the opposite sign and an omitted state-dependent gate. This is not a cosmetic discrepancy: it changes the fixed points, the interpretation of the “implicit activation target,” and the hyperparameter balance. The appendix claim that \(\lambda_S+\lambda_C<\lambda_V\) “prevents collapse” is not proved and does not follow from the stated loss. In particular, the covariance term is not generally a positive scalar magnitude penalty unless assumptions are imposed on the learned lateral operator, its sign, scaling, and approximation error.

There is a second missing term. Standard TeReL sets `detach_previous=False`. For one temporal pair,

\[
\frac{\partial (z_t-z_{t-1})^2}{\partial w}
=2(z_t-z_{t-1})
\left[x_t\phi'(a_t)-x_{t-1}\phi'(a_{t-1})\right].
\]

The proposition retains only the current-sample factor \(x_t\phi'(a_t)\) and treats \(p=z_{t-1}\) as fixed state. That formula applies only when the previous activation is detached. Without detachment, implementing the stated derivative requires retaining information sufficient to differentiate the preceding activation, not merely one scalar previous activation per neuron. In the batched chunk implementation, interior activations also participate in both neighboring pair losses. Thus the proposition does not establish the claimed three-state online implementation for the standard TeReL variant used in the headline experiments.

The paper needs one exact, indexed objective for each variant, with every `stop-gradient` operation, update order, reduction, centering operation, gate, and lateral-training rule shown. The propositions should then be rederived from those objectives. As written, the manuscript switches between an intuitive objective, a simplified scalar surrogate, and the actual code without clearly distinguishing them.

#### 3. The locality and online claims are stronger than either the derivation or experiments support

The main experiments use batches of 64, same-class chunks of 16, in-batch detached statistics, a dense learned \(D\times D\) lateral map, and (for TeReL) non-detached temporal pairs. This is a batched, layer-local procedure. It is not an experimental demonstration of an update that consumes a stream one item at a time using only three scalar state variables per neuron.

The claimed memory footprint also excludes important state. A neuron may keep three *dynamic scalars*, but the decorrelation mechanism adds \(D\) incoming lateral weights per unit and \(D^2\) auxiliary parameters per layer (8 million auxiliary parameters in the largest reported model even after the stated sparsity adjustment). Adam adds optimizer state for both forward and lateral parameters. This may still be a reasonable trade-off, but it is not characterized by “three scalar memory units per neuron” without a clear separation of dynamic state, parameters, optimizer state, minibatch storage, and communication.

The shifted-lateral proposition is not a mathematical proposition in its current form. “Compatible” and “informative” are undefined. For a linear lateral operator \(W\), a usable statement would require assumptions such as

\[
\|Wz_t-Wz_{t-1}\|\leq \|W\|_{\mathrm{op}}\|z_t-z_{t-1}\|,
\]

together with control of \(\|W\|_{\mathrm{op}}\), a definition of the target signal, and a bound on the resulting gradient error. Temporal coherence of the *input* alone does not ensure coherence of a learned representation with an uncontrolled Lipschitz constant. Empirically, the one-step lateral-shift ablation drops from 96.91% to 91.65% in the reported single run. This is evidence that the approximation can be damaging, not evidence that the headline setup is fully asynchronous or online.

The authors should either (a) narrow the principal claim to a batched layer-local learning procedure, or (b) implement and evaluate the actual streaming configuration: detached previous state, running rather than in-batch statistics, shifted/asynchronous lateral input if required, batch size one, and no stored computation graph. Memory, communication, wall-clock time, and energy should be measured if constrained-hardware advantages remain central claims.

#### 4. The experimental comparison is not controlled, and one backpropagation baseline is incorrectly implemented

The backpropagation commit cited by the paper (`2590863`) contains a separate defect. In `comparison/backprop_mnist.py`, the dispatch for model version 2 calls `forward_v1` rather than `forward_v2`. The nominal `3 × 500` model therefore ignores `fc4`, treats the 500 outputs of `fc3` as class logits, and is not the claimed three-hidden-layer, ten-class architecture. The corresponding row of Table 2 is invalid.

There are several additional protocol problems:

1. The code uses the official MNIST test split as the paper's “validation” set. The manuscript states that learning rates and TeReL hyperparameters were tuned using these results, leaving no untouched test set for the final comparisons.

2. Greedy TeReL receives 60 epochs *per layer* (120, 180, or 240 encoder epochs depending on depth) and then 60 head-training epochs, while the end-to-end backpropagation model receives 60 epochs. No examples-seen, FLOP, time, or energy matching is reported. A larger TeReL model therefore receives substantially more optimization work, despite the paper drawing a scale-dependent conclusion from accuracy alone.

3. TeReL's linear head reads the concatenation of all hidden layers; the backpropagation classifier reads only its final hidden layer. This is a meaningful architectural advantage, not merely a probe convention. A fair comparison requires the same readout access or a primary last-layer-only evaluation.

4. The Forward-Forward and Equilibrium Propagation values are borrowed from their original papers under different implementations and evaluation protocols. Forward-Forward is explicitly reported in a native headless format, whereas TeReL uses a trained linear head. These numbers are useful context, but they do not support a controlled comparative claim.

5. The main table reports only means even though seed-level values are available. The largest-layout TeReL–backpropagation difference varies substantially across seeds; uncertainty and paired differences should be reported. The RNN comparison and the displayed ablations appear to be single runs.

6. The ablation appendix does not support several statements made in the main text. In particular, the claim that removing any loss term caps accuracy at 70% is not shown in the appendix table or its run specifications. The appendix combines results produced from later code with the older, affected main-result commit without a result-to-commit map.

These issues require a new protocol: a train/validation/test split; corrected and matched architectures; identical readout access; matched tuning budgets; compute-aware accounting; repeated runs with uncertainty; and exact result provenance.

#### 5. The headline “self-supervised” evidence is label-supervised, synthetically ordered MNIST

For classification, positives are created by arranging samples into same-class chunks using ground-truth digit labels. The paper acknowledges that this is “effectively a supervised setup,” but the title, abstract, and conclusion continue to lead with self-supervised learning. The experiments demonstrate that label information can be injected through ordering and exploited by a local objective. They do not demonstrate self-supervised learning from an unlabeled sensory stream, robustness to imperfect temporal coherence, or recovery of naturally slow latent factors.

This distinction matters because the class-separated representations, selective neurons, and high linear-probe accuracy are expected consequences of placing same-label items next to one another. At least one genuinely unlabeled natural sequence benchmark is needed for the self-supervised claim. A controlled MNIST study should also vary coherence/noise, include random and augmentation-pair orderings, and compare to a matched supervised local contrastive or metric-learning objective so that the contribution of the specific TeReL regularizers is isolated.

#### 6. The relationship to SFA and prior local SSL is not positioned adequately

The paper describes TeReL as a bridge between VICReg and SFA, but the classical SFA objective already combines temporal slowness with zero-mean, unit-variance, and decorrelation constraints. Thus the three-part structure is not only analogous to SFA; at the objective level it substantially restates SFA with soft penalties and a neural parameterization. This does not make the work uninteresting, but it changes what must be identified as the contribution.

The related-work section also states that recent work has applied VICReg-like objectives greedily or layerwise, but gives no citation. The omission is important. [Siddiqui et al., *Blockwise Self-Supervised Learning at Scale*](https://openreview.net/forum?id=M2m618iIPk) explicitly evaluate blockwise Barlow Twins, VICReg, and SimCLR on ResNet-50/ImageNet. [Kompella et al., *Incremental Slow Feature Analysis*](https://doi.org/10.1162/NECO_a_00344) develop an online, covariance-free, Hebbian/anti-Hebbian incremental SFA procedure, including hierarchical settings. [Lipshutz et al., *A Biologically Plausible Neural Network for Slow Feature Analysis*](https://papers.nips.cc/paper_files/paper/2020/hash/ab73f542b6d60c4de151800b8abc0a6c-Abstract.html), which the manuscript cites but discusses only briefly, specifically derive an online neural SFA algorithm with local synaptic updates and evaluate it on naturalistic stimuli.

The manuscript must compare assumptions, state, locality, objective, update rule, computational cost, and empirical regime against these methods. TMLR does not require a method to be radically novel, but it does require accurate positioning and a clear finding of interest. At present it is unclear whether the substantive contribution is the particular detached variance surrogate, the learned lateral covariance approximation, the combination with deep greedy training, or simply a new implementation of soft SFA.

#### 7. The representation and recurrent analyses are descriptive and do not isolate TeReL

The representation section provides visually appealing diagnostics but little controlled evidence:

- class separation is evaluated after training on class-ordered streams and is not compared against backpropagation, TeReL-S, the untrained encoder, or an alternative local objective under the same projection and sampling procedure;
- “top selective neurons” are selected for selectivity, so observing selective units is not by itself evidence beyond the selection rule; a null distribution or effect-size comparison is needed;
- first-layer filters are selected by weight norm and described qualitatively as “rich” or “non-trivial” without a blinded metric or matched baseline;
- spectral and t-SNE projections can create apparent clusters and trajectories; the paper does not report stability across seeds, projection hyperparameters, or whether the side-by-side recurrent embeddings share one fitted coordinate system.

The MNIST-Rows result is also too limited to support the conclusion that TeReL “can support sequence models without backpropagation through time.” It reports one TeReL run and one backpropagation run, with TeReL worse on the primary metric (0.22345 versus 0.17569 MSE). Moreover, in the supplied sequence code the recurrent fusion module receives gradients from the next-row prediction loss in addition to the TeReL loss; it is not trained solely by the stated TeReL objective. This hybrid may still be temporally truncated and interesting, but the paper must describe it accurately, report multiple seeds, use a matched architecture, and separate the effects of the local prediction loss, temporal fusion, and TeReL regularization.

#### 8. Reproducibility is insufficient for a journal submission

The repository is useful but does not presently define a reproducible paper artifact. There is no pinned dependency/environment file, no hardware/software specification, no single command or manifest for each table row, no immutable mapping from every result to a commit/configuration/checkpoint, and no automated test that verifies all intended layers are updated. The README explicitly notes that the current code has changed since the main experiments. The sequence appendix says that the exact repository state is recorded in the README, but the README lists two commits without mapping them to the reported model, seed, command, or checkpoint.

The current `train.py` configuration is not a reproduction of the headline table. The paper's central algorithm, pseudocode, and code also differ in loss sign, variance gating, detachment, batch reduction, activation application in the historical experiment code, and lateral-target updates. These differences make independent verification unusually difficult.

## Would at least some individuals in TMLR's audience be interested in knowing the findings of this paper?

**Yes, conditionally.**

Researchers working on local credit assignment, SFA, online representation learning, and learning on constrained hardware could be interested in a careful evaluation of a soft, VICReg-inspired SFA rule with detached statistics and a learned lateral covariance signal. The negative findings—optimizer sensitivity, failure of shifted lateral signals, and the importance of variance/decorrelation terms—could also be useful if established under a valid protocol.

My negative recommendation is therefore not based on a lack of perceived significance or novelty. It is based on the gap between the submission's claims and the mathematical and empirical evidence. Once the implementation defects are corrected, it is possible that a narrower and well-supported paper would meet TMLR's interest criterion.

## Requested changes

### Critical to an acceptance recommendation

1. **Rerun every central experiment from corrected, immutable code.** Fix optimizer registration; verify and log that each encoder and lateral layer changes; fix the `3 × 500` backpropagation dispatch; and regenerate the main table, ablations, representation analyses, and parameter claims. Do not reuse results from affected commits.

2. **Replace the current theoretical section with a faithful derivation.** Define exact TeReL and TeReL-S objectives, indices, state updates, stop-gradient locations, variance gate/sign, reductions, and lateral optimization. Derive the full weight gradient for detached and non-detached temporal references. Remove or weaken statements that are only design intuitions. Provide explicit assumptions and an error bound for the shifted-lateral approximation, or label it a heuristic.

3. **Resolve the online/locality mismatch.** Either evaluate a genuinely streaming implementation using bounded forward state and no stored computation graph, or consistently call the evaluated method batched and layer-local. Account separately for dynamic state, model parameters, lateral parameters, optimizer state, batch storage, and communication. Measure hardware-relevant costs if hardware advantages are claimed.

4. **Use a valid, matched evaluation protocol.** Preserve an untouched test split; use the same readout access and comparable architectures; match or report training compute/examples; tune all methods under comparable budgets; run at least five seeds for central and recurrent results; and report uncertainty and result provenance.

5. **Support or narrow the self-supervised claim.** Add at least one natural unlabeled sequential dataset and relevant online/local baselines, or retitle and reframe the paper as a supervised class-ordering study. Include ordering/coherence controls that distinguish TeReL from generic local metric learning.

6. **Reposition the contribution against SFA and local SSL.** Discuss original SFA, Incremental SFA, Bio-SFA, and blockwise VICReg/Barlow-Twins work concretely. State what TeReL contributes beyond softening classical SFA constraints and applying them greedily to an MLP.

7. **Create a reproducible paper package.** Pin dependencies; provide exact commands/config files, seeds, hardware, checkpoints or logs, and a result manifest; add tests that assert all intended parameter groups update; and make the paper's equations and code agree.

These requests amount to a new experimental and mathematical evaluation. They are not realistically resolvable through a small revision or author-response clarification.

### Non-critical but important for clarity

1. Report quantitative representation comparisons against matched random, backpropagation, TeReL-S, SFA/Bio-SFA where feasible, and other local-objective controls rather than relying primarily on selected visualizations.

2. Clarify that the lateral map is discarded only at inference, while its parameters, training state, and communication remain part of training cost.

3. Explain the recurrent fusion training path accurately, including which modules receive next-row prediction gradients and which temporal dependencies are detached.

4. Replace “Hebbian style” with a precise statement of which pre- and postsynaptic quantities each update uses. A product-form gradient is not sufficient by itself to establish biological plausibility.

5. Distinguish “validation” from the official MNIST test split throughout, and reserve “test” for an untouched final evaluation.

6. Expand the limitations section to cover label-derived ordering, the dense lateral map, absence of measured hardware benefits, comparison mismatch, single-dataset generalization, and the dependence of the online claim on detachment.

## Questions for the authors

1. Were all Table 2 TeReL/TeReL-S values produced with commit `b39b73f` exactly as stated? If not, what immutable commit, configuration, and logs produced each value?

2. Can the authors provide evidence that parameters after the first encoder layer changed in the reported main runs? The cited code makes this appear impossible.

3. Which of TeReL and TeReL-S is intended to satisfy the three-scalar online-state claim? How is the derivative through an undetached previous activation computed after weights have been updated, without retaining the preceding layer input or computation graph?

4. What is the exact variance loss? Is the positive \((z-m)^2\) term in the proposition intentional, or should it be the gated negative term used by the code? What formal statement supports the claim that a coefficient inequality prevents collapse?

5. What quantity does the lateral map approximate at each time: covariance, off-diagonal covariance, a gradient of squared covariance, or a lagged cross-covariance? How does lateral approximation error affect the main-network gradient?

6. Why is access to all hidden layers used for the TeReL head but not the backpropagation baseline? What are the repeated-run results under last-layer-only probes and identical readouts?

7. How were hyperparameters selected without evaluating on the official MNIST test split? Is there an unreported validation split?

8. Which exact sequence configuration produces 0.22345 MSE? Does the temporal fusion module receive prediction-loss gradients, and if so, why does the manuscript say that both the preprocessor and recurrent aggregator are trained using the TeReL objective while only the linear head is trained for prediction?

## Minor presentation issues

The manuscript requires a careful technical edit. Examples include:

- “Discriminative and perceptual **learnig** on MNIST”;
- “TeReL is **a a** greedy training procedure that **constrats** itself...”;
- “chunk_size=32 **achieves achieves** 93.98% accuracy”;
- the sentence pointing to the MNIST-Rows appendix appears twice consecutively;
- phrases such as “there should be plenty of time,” “the main move,” “we go over,” “holds in principle,” “compatible,” and “informative” are too informal or underspecified for technical claims;
- “validation,” “test,” “stream,” “chunk,” “batch,” “online,” “parallelizable,” “per-neuron,” and “local” are used in ways that blur materially different settings;
- Proposition numbering and labels use theorem-style identifiers for several statements that are definitions, algebraic observations, or heuristics rather than substantive propositions;
- the pseudocode omits the lateral-map training rule, loss reductions, detachments, chunk-boundary behavior, and the order in which statistics are updated, so it is not a complete algorithm specification.

Writing quality alone would not determine my recommendation. Here, however, imprecise writing masks important differences between the proved surrogate, the implemented method, and the evaluated configurations.

## Broader impact concerns

I do not identify a material ethical or societal risk that requires a separate broader-impact statement. The hardware and biological-plausibility discussion should nevertheless be narrowed to evidence. Claims of energy efficiency, neuromorphic suitability, or biological plausibility can mislead readers when no hardware measurements or mechanistic biological validation are provided.

## Final assessment

**Recommendation: Reject.**

The paper asks an interesting question and contains the seed of a potentially useful empirical study. In its current form, however, the headline deep-learning results were generated with code that only optimizes the first layer; one backpropagation comparison is misimplemented; the central gradient proposition has the wrong variance sign and omits the non-detached temporal derivative; and the experiments do not instantiate the claimed online self-supervised setting. These are decisive soundness issues under TMLR's primary claims-and-evidence criterion.

I would be willing to review a substantially revised resubmission built on corrected experiments, an exact derivation, a controlled evaluation, and narrower claims. The result may ultimately be a useful paper, but the present evidence cannot support acceptance.
