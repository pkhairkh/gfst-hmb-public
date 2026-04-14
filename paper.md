# MPUC and Sparse-Support Centroid Recall for Exact Long-Context Memory
## Grassmann-Flag State Tracking with a Multirate Predictive-Unification Core and Multi-Block Sparse Associative Readout

**Abstract**  
We introduce a long-context memory architecture that treats exact recall as a **sparse associative reconstruction problem** rather than as coarse block routing alone. The architecture combines a local Grassmann-Flag State Tracker (GFST) and a linear-time MPUC backbone with an external exact-token memory organized as block-local lookup tables (LUTs). Each block stores exact token IDs together with a small learned set of inner memory cells. Tokens are written into this memory through a **top-2 sparse assignment**, producing phase-bound cell summaries that preserve both content and local occurrence information. At read time, a coarse geometric router retrieves a small set of candidate blocks, and a **multi-head centroid readout** selects significant block-cell supports across those blocks by robust standardization and Benjamini-Hochberg false-discovery-rate gating. Final token scores combine the base token score with support mass and a centroid recall term under an adaptive confidence gate.

The central claim is not that a single geometric block key is sufficient for exact memory, but that **exact-token LUT blocks plus sparse-support centroid recall** can recover occurrence-specific information without storing full hidden states for every token. The resulting memory scales linearly in context length, keeps exact-token storage cheap, and delegates the main discrimination burden to sparse write-time support formation and sparse multi-block readout. We also describe the current recovery-stage training regime: a frozen Qwen3-1.7B teacher, an MPUC-first student, selective centroid replay, and auxiliary losses for compactness, sharpness, coherence, cardinality, and head specialization. The paper is forward-looking: it states the current canonical architecture and the empirical program required to validate it at million-token scale.

---

## 1. Introduction

Long-context language modeling has two intertwined requirements: efficient streaming state updates and the ability to recover **exact, occurrence-specific content** far outside the local recurrent horizon. Linear-time multirate recurrent cores such as MPUC offer an attractive substrate for the first requirement [1], while Grassmannian subspace tracking provides compact geometric summaries of local context [2, 3]. But exact long-context memory is not solved by linear recurrence alone. A bounded recurrent state can summarize, but it cannot by itself preserve arbitrarily many exact token occurrences over million-token contexts.

This observation changes how the memory problem should be formulated. Exact recall is not merely a question of retrieving a relevant block from a coarse index. It is a question of **binding a partial cue to a sparse support that identifies the correct occurrence**. In neuroscience terms, dense overlapping representations support semanticization but degrade episodic specificity as the number of stored examples grows [4-7]. In mathematics, exact recovery in inverse problems depends on sparse support and low mutual coherence rather than on diffuse overlap [8-10]. In astrophysical source detection, low-signal recovery improves when scores are variance-stabilized and thresholded by significance rather than by raw amplitude [11-13]. In associative systems from statistical and quantum physics, metastable recall states are stabilized by internal fields and self-interactions [14, 15].

We therefore propose **GFST-HMB as sparse-support centroid memory over exact-token LUT blocks**. The architecture has three layers of function:

- a **local streaming state substrate** built from MPUC and a Grassmann-flag tracker,
- a **coarse block router** built from compact GFST-derived block summaries,
- a **sparse associative readout** that operates over learned inner cells inside retrieved exact-token blocks.

The important shift is that the coarse geometric router is no longer treated as the sole discriminative mechanism. A minimal mean-plus-top-eigenvector block key remains a valid coarse routing baseline, but exact discrimination is delegated to a richer centroid memory with five ingredients:

1. sparse top-2 token-to-cell write binding,
2. phase-bound key/value storage to encode local occurrence,
3. significance-gated cell selection across multiple retrieved blocks,
4. a small self-field bias that stabilizes salient cells,
5. confidence-gated fusion of centroid recall into token scores.

The main hypothesis is:

> **Exact long-context recall requires sparse support recovery plus occurrence binding.**

Under this view, failures of exact recall are expected when write-side support is too diffuse, even if coarse routing and read-side sparsity appear healthy. The architecture described here is designed to attack that bottleneck directly while avoiding full per-token hidden-state caching.

Our contributions are:

1. an explicit **MPUC local backbone** with multirate recurrent bands, cross-band gating, semantic adaptation, and event-sensitive slow updates,
2. a **canonical sparse-support centroid memory** layered on top of GFST/MPUC,
3. a mathematically explicit write/read formulation for block-local cells and multi-block centroid recall,
4. a recovery-stage training recipe centered on selective centroid replay and sparse-support regularization,
5. an empirical program that prioritizes **support-formation health** before full end-to-end exact-recall claims.

---

## 2. MPUC Backbone and GFST State

The local backbone is not the whole memory system, but it provides the streaming state on which the memory system is built.

### 2.1 Multirate Predictive-Unification Core

The local backbone is a **multirate predictive-unification core (MPUC)** rather than a homogeneous deep recurrent stack. Its role is to maintain a bounded, incrementally updated language state with simultaneous short-, mid-, and long-timescale traces before any explicit memory read.

Let \(\bar{h}_t \in \mathbb{R}^{d}\) denote the normalized residual stream at token \(t\). A shallow causal multiresolution front-end applies three depthwise-convolution branches with distinct receptive fields:

\[
d_t^{(f)} = \operatorname{PW}_f(\operatorname{DWConv}_{3,1}(\bar{h}_{\le t})),
\qquad
d_t^{(m)} = \operatorname{PW}_m(\operatorname{DWConv}_{5,2}(\bar{h}_{\le t})),
\qquad
d_t^{(s)} = \operatorname{PW}_s(\operatorname{DWConv}_{9,4}(\bar{h}_{\le t})).
\]

These branches feed three interleaved recurrent bands:

\[
 h_t^{f} = (1-g_t^{f}) \odot h_{t-1}^{f} + g_t^{f} \odot \tanh(\xi_t^{f}),
\]
\[
 h_t^{m} = (1-g_t^{m}) \odot h_{t-1}^{m} + g_t^{m} \odot \tanh(\xi_t^{m}),
\]
\[
 h_t^{s} = (1-g_t^{s} u_t^{s}) \odot h_{t-1}^{s} + (g_t^{s} u_t^{s}) \odot \tanh(\xi_t^{s}),
\]

where the drives \(\xi_t^{(\cdot)}\) contain both branch-local evidence and low-rank cross-band terms:

\[
\xi_t^{f} = d_t^{(f)} + P_m h_{t-1}^{m} + P_s h_{t-1}^{s},
\]
\[
\xi_t^{m} = d_t^{(m)} + Q_f h_t^{f} + Q_s h_{t-1}^{s},
\]
\[
\xi_t^{s} = d_t^{(s)} + R_f h_t^{f} + R_m h_t^{m}.
\]

The slow band is modulated by an event/boundary controller

\[
u_t^{s} = \sigma\!\left(\operatorname{MLP}\!\left[
\operatorname{mean}(g_t^{f}),
\operatorname{mean}(g_t^{m}),
\operatorname{rms}(\xi_t^{m} - h_{t-1}^{m}),
\operatorname{rms}(\xi_t^{s} - h_{t-1}^{s})
\right]\right),
\]

so long-timescale state updates are strongest at moments of elevated novelty or boundary evidence instead of being uniformly forced at every token.

MPUC also includes a low-rank semantic adapter that reconditions token meaning on the current mid/slow state:

\[
\tilde{h}_t
=
W_o [h_t^{f}; h_t^{m}; h_t^{s}]
+
U \,\operatorname{diag}\!\bigl(\phi([h_t^{m}; h_t^{s}])\bigr)\, V^\top \bar{h}_t .
\]

This makes the local backbone explicitly multirate, context-conditioned, and boundary-sensitive. GFST operates on this state, not on a raw token embedding stream.

### 2.2 Token State and Reduced Geometric Coordinates

Let \(\tilde{h}_t \in \mathbb{R}^{d}\) be the backbone hidden state produced by a linear-time recurrence. In the current project instantiation, this recurrence is MPUC [1]. We define a reduced geometric state

\[
x_t = \mathrm{RoPE}(W_x \tilde{h}_t) \in \mathbb{R}^{r},
\]

where \(W_x \in \mathbb{R}^{r \times d}\) and \(r\) is a reduced dimension (e.g. \(r = 64\)).

### 2.3 Residual-Aware Grassmann Update

For the largest flag level of rank \(k\), the tracker maintains an orthonormal basis

\[
U_t \in \mathbb{R}^{r \times k}, \qquad U_t^\top U_t = I_k.
\]

The current heuristic residual-aware update is

\[
a_t = U_{t-1}^\top x_t,\qquad
p_t = U_{t-1} a_t,\qquad
r_t = x_t - p_t,
\]

\[
v_t = \frac{r_t}{\sqrt{\|x_t\|_2^2 + \varepsilon_{\mathrm{upd}}^2}},
\]

\[
\widetilde{U}_t
=
(1-\alpha)U_{t-1}
+ \alpha\,x_t a_t^\top
+ \alpha\,v_t e_k^\top,
\qquad
U_t = \operatorname{cQR}(\widetilde{U}_t),
\]

where \(e_k\) is the final basis vector and \(\operatorname{cQR}\) denotes reduced QR with deterministic sign fixing. This update is not a claim of Grassmann-optimality; it is a bounded, residual-aware tracker that responds to novelty while remaining cheap and stable.

### 2.4 Flag Hierarchy

To represent multiple local time scales, we maintain a nested flag

\[
V_t^{(1)} \subset V_t^{(2)} \subset \cdots \subset V_t^{(m)} \subseteq \mathbb{R}^{r},
\]

with dimensions \(k_1 < k_2 < \cdots < k_m\). The largest level is updated directly from \(x_t\). Smaller levels are updated in the coordinates of the largest level:

\[
x_t^{\mathrm{proj}} = U_t^{(m)\top} x_t,
\qquad
U_t^{(i)} = U_t^{(m)} V_t^{(i)}.
\]

The top level captures fast novelty; smaller levels act as slower geometric summaries.

### 2.5 Basis-Invariant Local Features

For each level \(i\), define the projector

\[
P_t^{(i)} = U_t^{(i)} U_t^{(i)\top}.
\]

The normalized projection energy and incremental flag mass are

\[
\rho_t^{(i)} = \frac{x_t^\top P_t^{(i)} x_t}{\|x_t\|_2^2 + \varepsilon},
\qquad
g_t^{(i)} = \rho_t^{(i)} - \rho_t^{(i-1)},
\qquad
\rho_t^{(0)} := 0.
\]

The local geometric feature vector is

\[
\phi_t^{\mathrm{local}}
=
\bigl[
\rho_t^{(1)}, g_t^{(1)},
\rho_t^{(2)}, g_t^{(2)},
\ldots,
\rho_t^{(m)}, g_t^{(m)}
\bigr].
\]

These features are basis-invariant and summarize how much of the current token state is captured at different geometric scales.

### 2.6 Local Fusion

The local geometric features modulate the backbone state by a gated residual:

\[
h_t' = \tilde{h}_t + \sigma(W_g \phi_t)\odot (W_f \phi_t),
\]

where \(\phi_t\) will later include both local and memory-derived features. The crucial point is that this local state remains **bounded and streaming**. By design, it is not sufficient for exact long-range recall on its own.

---

## 3. Sparse-Support Centroid Memory

The memory system augments the local GFST state with a sparse associative layer over exact-token blocks.

### 3.1 Coarse Block Routing and Exact-Token LUT Blocks

We partition the context into non-overlapping blocks of length \(B\). For each block \(b\), the GFST produces a compact block summary \(U_b\) and a router key \(k_b^{\mathrm{route}}\). The router may be implemented with a minimal key such as a projection of the block mean together with a canonical leading direction, but this coarse key is only the **first routing stage**, not the primary exact-discrimination mechanism.

For a token query at time \(t\), the router produces a shortlist

\[
\mathcal{R}_t \subset \{1,\ldots,N\},
\qquad
|\mathcal{R}_t| = R,
\]

using approximate nearest-neighbor search over the block keys.

Each retrieved block is backed by an exact-token lookup table (LUT) containing:

- exact token IDs,
- the token's top-2 cell memberships,
- quantized token-to-cell membership weights,
- block-local cell summaries.

The exact block store is external to the local recurrent state and is accessed through a bounded exact cache. This separation is essential: the recurrent backbone summarizes, while the LUT blocks preserve exact symbolic content.

### 3.2 Sparse Write Formation

Each block contains \(C\) learned inner cells. In the current canonical configuration:

\[
C = 16,\quad
H = 4,\quad
d_k = 32,\quad
d_v = 16.
\]

Write-time cell formation starts from a phase-augmented token state

\[
\widehat{h}_t = h_t' + P_{\mathrm{phase}}(\psi_t),
\]

where \(\psi_t\) is a deterministic local-offset code inside the block. This phase injection helps separate repeated content that appears at different positions.

Cell-assignment logits are

\[
\ell_t = W_a \widehat{h}_t \in \mathbb{R}^{C}.
\]

The canonical write assignment is a **top-2 sparse simplex projection**

\[
a_t = \operatorname{SparseTop2}\!\left(\frac{\ell_t}{\tau_a}\right),
\qquad
a_t \in \Delta^{C-1},
\qquad
\|a_t\|_0 \le 2,
\]

with temperature \(\tau_a\). Thus each token is written to at most two cells.

We then form key/value write vectors

\[
k_t = W_k \widehat{h}_t,
\qquad
v_t = W_v \widehat{h}_t,
\]

and bind them to local offset through a phase rotation

\[
\widetilde{k}_t = R(\varphi_t) k_t,
\qquad
\widetilde{v}_t = R(\varphi_t) v_t.
\]

This is the occurrence-binding step: it separates "what" from "where/when" without storing full hidden states for every token [7, 11].

For block \(b\) and cell \(c\), the stored cell summaries are weighted means

\[
K_{b,c}
=
\frac{\sum_{t \in b} a_{t,c}\,\widetilde{k}_t}
     {\sum_{t \in b} a_{t,c} + \varepsilon},
\qquad
V_{b,c}
=
\frac{\sum_{t \in b} a_{t,c}\,\widetilde{v}_t}
     {\sum_{t \in b} a_{t,c} + \varepsilon},
\]

with cell mass

\[
n_{b,c} = \sum_{t \in b} a_{t,c}.
\]

The exact token store itself remains cheap: in the current implementation, token ID plus top-2 cell IDs and top-2 quantized weights require only \(O(L)\) storage and are small compared with multi-layer GFST block summaries.

### 3.3 Self-Field and Significance-Gated Multi-Block Readout

The centroid readout operates only on retrieved blocks \(b \in \mathcal{R}_t\), but it can select cells across **multiple** blocks simultaneously.

For each head \(h \in \{1,\ldots,H\}\), define a query

\[
q_{t,h} = W_q^{(h)} x_t \in \mathbb{R}^{d_k}.
\]

Each retrieved block-cell pair \((b,c)\) receives a raw score

\[
s_{t,h,b,c}
=
\frac{\langle q_{t,h}, K_{b,c}\rangle}{\sqrt{d_k}}
\lambda_{\mathrm{blk}}\,\beta_{t,b}
\lambda_{\mathrm{self}}\,u_{b,c},
\]

where \(\beta_{t,b}\) is the coarse router score for block \(b\), and

\[
u_{b,c}
=
\log(1+n_{b,c})
\;-\;
\frac{1}{C}\sum_{c'=1}^{C}\log(1+n_{b,c'})
\]

is a centered self-field term. This term is small but important: it biases the dynamics toward cells with stable local support, echoing the stabilizing role of self-interactions in associative systems [13-15].

Because raw scores are not directly comparable in noisy, low-count regimes, we use robust standardization:

\[
\bar{s}_{t,h,b,c}
=
s_{t,h,b,c}

- \operatorname{med}_{(b',c') \in \mathcal{R}_t}(s_{t,h,b',c'}),
\]

\[
\sigma_{t,h}
=
1.4826 \cdot \operatorname{MAD}_{(b',c') \in \mathcal{R}_t}(\bar{s}_{t,h,b',c'})
\varepsilon,
\]

\[
z_{t,h,b,c}
=
\frac{\bar{s}_{t,h,b,c}}{\sigma_{t,h}}
\;-\;
\lambda_{\mathrm{cnt}}\log(1+n_{b,c}).
\]

We then convert these standardized scores into one-sided tail probabilities

\[
p_{t,h,b,c} = 1 - \Phi(z_{t,h,b,c}),
\]

and apply Benjamini-Hochberg false-discovery-rate control at level \(q_{\mathrm{FDR}}\) to select a sparse support set

\[
\mathcal{S}_{t,h} \subseteq \{(b,c): b \in \mathcal{R}_t\}.
\]

If too few cells survive, we fall back to a small minimum support per head; in the current configuration this minimum is \(2\), with a cap of \(8\) cells per head. The selected support weights are

\[
\alpha_{t,h,b,c}
=
\frac{\exp(z_{t,h,b,c}/\tau_{\mathrm{sel}})}
     {\sum_{(b',c') \in \mathcal{S}_{t,h}} \exp(z_{t,h,b',c'}/\tau_{\mathrm{sel}})},
\qquad
(b,c) \in \mathcal{S}_{t,h}.
\]

### 3.4 All Tokens Stay in Play

An important design choice is that centroid selection does **not** hard-prune tokens. All tokens in retrieved blocks remain valid candidates:

\[
\mathcal{C}_t
=
\bigcup_{b \in \mathcal{R}_t}
\{(b,j): 1 \le j \le B_b^{\mathrm{prefix}}\}.
\]

The centroid subsystem modulates token scores; it does not erase candidate coverage.

Let candidate \(i\) correspond to occurrence \((b(i), j(i))\). Each token occurrence stores up to two cell IDs and weights:

\[
\bigl(c_{i,1}, w_{i,1}\bigr),\;
\bigl(c_{i,2}, w_{i,2}\bigr).
\]

Its centroid support mass is

\[
m_i
=
\sum_{r=1}^{2} w_{i,r}
\sum_{h=1}^{H}
\alpha_{t,h,b(i),c_{i,r}}\,
\mathbf{1}\!\left[(b(i),c_{i,r}) \in \mathcal{S}_{t,h}\right].
\]

Its centroid recall term is a multi-seed log-sum-exp:

\[
r_i
=
\log
\sum_{h=1}^{H}
\sum_{(b,c)\in\mathcal{S}_{t,h}}
\alpha_{t,h,b,c}\,
\exp\!\left(
\frac{\langle g_{i,h}, V_{b,c}\rangle}{\sqrt{d_v}}
\right),
\]

where

\[
g_i = W_{\mathrm{cent}} e(\mathrm{id}_i),
\]

and \(e(\mathrm{id}_i)\) is the token embedding of the candidate token ID.

The final candidate score is

\[
\mathrm{score}_i
=
\mathrm{base}_i

+ (\gamma_m\,\mathrm{conf}_t)\,m_i
+ (\gamma_r\,\mathrm{conf}_t)\,r_i.
\]

Here \(\mathrm{base}_i\) is the baseline token score within the retrieved exact blocks, and the centroid contribution is confidence-gated:

\[
\mathrm{conf}_t
=
\mathrm{conf}^{\mathrm{ent}}_t \cdot \mathrm{conf}^{\mathrm{sig}}_t.
\]

\(\mathrm{conf}^{\mathrm{ent}}_t\) decreases when selected support entropy is diffuse, while \(\mathrm{conf}^{\mathrm{sig}}_t\) increases when the retained support is statistically significant. This mechanism suppresses centroid influence when the readout is uncertain.

### 3.5 Copy Decision and Complexity

The readout returns the highest-scoring candidate

\[
i^* = \arg\max_{i \in \mathcal{C}_t} \mathrm{score}_i.
\]

The model then chooses between generation and copy:

\[
p(y \mid t)
=
\pi_t^{\mathrm{copy}}\,\delta_{\mathrm{id}_{i^*}}(y)
\;+\;
\bigl(1-\pi_t^{\mathrm{copy}}\bigr)\,\mathrm{softmax}(\mathrm{logits}_{\mathrm{vocab}})_y.
\]

Storage and compute scale as follows:

- local GFST state: streaming and bounded,
- coarse router memory: \(O(N)\) block summaries and router keys,
- exact-token LUT store: \(O(L)\),
- block-cell summaries: \(O(N C(d_k+d_v))\),
- routed centroid scoring: \(O(HRCd_k)\),
- token scoring over retrieved exact blocks: \(O(RBd_v)\) in the worst case.

The key systems point is that the exact-token store itself is not the dominant memory cost. The main persistent cost remains the multi-layer GFST block-summary hierarchy; the centroid exact store is comparatively cheap.

---

## 4. Training Regime

The architecture can in principle be trained in several ways. The current project instantiation uses a **teacher-guided recovery regime** that is explicit about the transition from a frozen upstream teacher to the sparse-support centroid student.

### 4.1 Recovery Setting

The current recovery stage uses:

- a frozen **Qwen3-1.7B** model as teacher,
- the same teacher tokenizer and vocabulary as the student anchor,
- an **MPUC-first student** augmented with GFST-HMB sparse-support centroid memory.

The teacher provides logit supervision and a stable vocabulary interface; it is not itself the memory model being proposed.

### 4.2 Main Objective

For a training window, the full recovery objective is

\[
\mathcal{L}
=
\lambda_{\mathrm{KL}}\mathcal{L}_{\mathrm{distill}}
\;+\;
\lambda_{\mathrm{CE}}\mathcal{L}_{\mathrm{next}}
\;+\;
\lambda_{\mathrm{copy}}\mathcal{L}_{\mathrm{copy}}
\;+\;
\lambda_{\mathrm{ptr}}\mathcal{L}_{\mathrm{pointer}}
\;+\;
\lambda_{\mathrm{ret}}\mathcal{L}_{\mathrm{retrieval}}
\;+\;
\mathcal{L}_{\mathrm{centroid}}.
\]

The first five terms are standard teacher-guided recovery terms. The centroid-specific regularizer is

\[
\mathcal{L}_{\mathrm{centroid}}
=
\lambda_{\mathrm{cmp}}\mathcal{L}_{\mathrm{compact}}
\;+\;
\lambda_{\mathrm{shr}}\mathcal{L}_{\mathrm{sharp}}
\;+\;
\lambda_{\mathrm{coh}}\mathcal{L}_{\mathrm{coherence}}
\;+\;
\lambda_{\mathrm{card}}\mathcal{L}_{\mathrm{cardinality}}
\;+\;
\lambda_{\mathrm{ov}}\mathcal{L}_{\mathrm{overlap}}
\;+\;
\lambda_{\mathrm{div}}\mathcal{L}_{\mathrm{diversity}}.
\]

### 4.3 Sparse-Support Auxiliary Terms

**Compactness.** Tokens assigned to the same cell should concentrate around that cell's mean key:

\[
\mathcal{L}_{\mathrm{compact}}
=
\frac{
\sum_{b}\sum_{t \in b}\sum_{c}
a_{t,c}\,\|\widetilde{k}_t - K_{b,c}\|_2^2
}{
\sum_{b}\sum_{t \in b}\sum_{c} a_{t,c} + \varepsilon
}.
\]

**Sharpness.** Write assignments should remain sparse rather than diffuse:

\[
\mathcal{L}_{\mathrm{sharp}}
=
\frac{1}{|\mathcal{T}|}
\sum_{t}
\frac{-\sum_c a_{t,c}\log(a_{t,c}+\varepsilon)}{\log C}.
\]

**Coherence.** Active cell keys within a block should be as decorrelated as possible:

\[
\mathcal{L}_{\mathrm{coherence}}
=
\frac{1}{|\mathcal{B}|}
\sum_b
\frac{1}{|A_b|(|A_b|-1)}
\sum_{\substack{c,c' \in A_b\\c\neq c'}}
\bigl(\widehat{K}_{b,c}^\top \widehat{K}_{b,c'}\bigr)^2,
\]

where \(A_b\) is the active-cell set and \(\widehat{K}_{b,c}\) are normalized cell keys.

**Cardinality.** A block should not activate too many materially occupied cells. Let

\[
\mu_{b,c}
=
\frac{n_{b,c}}{\sum_{c'}n_{b,c'}+\varepsilon},
\qquad
\mathrm{supp}_{\theta}(b)
=
\{c:\mu_{b,c}\ge \theta_s\}.
\]

Then

\[
\mathcal{L}_{\mathrm{cardinality}}
=
\frac{1}{|\mathcal{B}|}
\sum_b
\max\!\left(0,\;|\mathrm{supp}_{\theta}(b)|-\kappa\right)^2.
\]

In the current canonical configuration, \(\theta_s = 0.10\) and \(\kappa = 3.5\), encouraging a regime with roughly \(2\)-\(4\) materially active cells per block.

**Head overlap.** Different centroid heads should not repeatedly select the same block-cell support:

\[
\mathcal{L}_{\mathrm{overlap}}
=
\frac{1}{|\mathcal{T}|}
\sum_t
\frac{1}{H(H-1)}
\sum_{h\neq h'}
\sum_{(b,c)}
\alpha_{t,h,b,c}\,\alpha_{t,h',b,c}.
\]

**Head diversity.** Head queries should not collapse geometrically:

\[
\mathcal{L}_{\mathrm{diversity}}
=
\frac{1}{|\mathcal{T}|}
\sum_t
\frac{1}{H(H-1)}
\sum_{h\neq h'}
\cos^2(q_{t,h}, q_{t,h'}).
\]

A weak long-horizon occupancy prior can be added, but it is no longer the main shaping mechanism. The primary pressure is sparse support, low coherence, and low overlap.

### 4.4 Selective Centroid Replay

Before normal training, the model runs a short **centroid replay warm-up**. During this stage, only centroid-role parameters are updated, and replay focuses on ambiguous positions rather than uniform rehearsal.

Define a replay mask

\[
M_t
=
\mathbf{1}
\Bigl[
\Delta_t^{\mathrm{ptr}} < \delta_{\mathrm{ptr}}
\;\lor\;
H_t^{\mathrm{sel}} > \delta_H
\;\lor\;
\mathrm{conf}_t < \delta_{\mathrm{conf}}
\Bigr].
\]

In the current configuration,

\[
\delta_{\mathrm{ptr}} = 0.20,\qquad
\delta_H = 0.60,\qquad
\delta_{\mathrm{conf}} = 0.15.
\]

Replay optimizes pointer and centroid-support terms on these focused positions, falling back to all valid positions only when no positions satisfy the ambiguity criteria. This replay is intended to stabilize sparse support formation before broader holistic training.

---

## 5. Discussion and Open Questions

The central unresolved issue is no longer whether a single block key is discriminative enough in isolation. The primary risk is now:

> **write-side support ambiguity**.

If the write path spreads identity across too many cells, then even a sparse readout cannot produce occurrence-specific exact recall. A model can appear selective at read time while remaining diffuse at write time; such a system tends to improve broad associative spread while failing to sharpen top-rank exact discrimination.

This diagnosis is consistent with three complementary literatures:

- **neuroscience:** episodic specificity depends on sparse, pattern-separated engrams and phase-sensitive temporal coding [4-7],
- **mathematics:** exact recovery depends on sparse support and low mutual coherence [8-10],
- **astrophysics and inverse problems:** significance detection under low-count noise improves after variance stabilization and FDR control [11-13],
- **physics:** associative recall becomes more stable when internal self-fields reshape the energy landscape [14, 15].

Within this framework, several design decisions become principled:

- sparse top-2 write binding is a support-identification mechanism,
- phase modulation binds local occurrence without full hidden-state caching,
- coherence regularization improves recoverability of cell supports,
- FDR-gated readout prevents diffuse score tails from dominating selection,
- self-field bias stabilizes recurring supports,
- selective replay aligns training effort with ambiguous or weakly separated positions.

The main open questions are:

1. whether write-side sparsity can be maintained under longer holistic training,
2. whether sparse-support health translates into improved pointer top-1 and end-to-end exact recall,
3. how much block router richness is actually needed once centroid memory is functioning well,
4. whether exact-token LUT blocks plus sparse centroid summaries are sufficient at million-token scale without full hidden-state storage.

---

## 6. Empirical Program

The empirical program should follow a staged ladder. The goal is to validate the **mechanism** before claiming end-to-end exact recall.

### 6.1 Stage A: Tiny Signal Probes

Run very short warm-start probes on synthetic exact-recall windows. The first gate is support-formation health, not end-task score.

Track:

- `centroid_active_cells_per_block`,
- `centroid_selected_cells_per_head_mean`,
- centroid assignment entropy,
- compactness,
- coherence,
- cardinality,
- pointer loss.

Primary success condition:

- block-level active support collapses from the diffuse regime toward roughly \(2\)-\(4\) cells per block,
- head-level selection remains sparse,
- candidate coverage does not collapse.

### 6.2 Stage B: Short Warm-Start Pointer/Copy Probes

Only if Stage A looks healthy, run a short warm-start probe with the full pointer and copy heads active. Compare against a token-only baseline using:

- gold-in-candidate rate,
- pointer top-1 when gold is in candidate,
- gold MRR,
- top-10 and top-32 exact-ranking metrics.

The criterion is not only lower training loss, but **better exact ranking without candidate-coverage regression**.

### 6.3 Stage C: Long-Context Exact Recall

Only after Stage A and Stage B pass should the model be pushed toward larger contexts and harder tasks:

1. synthetic million-token needle recall,
2. exact citation/copy-paste style benchmarks,
3. mixed tasks requiring both exact recall and coarse semantic structure.

The key ablations are:

- number of retrieved blocks \(R\),
- cells per block \(C\),
- write sparsity level,
- significance threshold via \(q_{\mathrm{FDR}}\),
- self-field strength,
- phase binding on/off,
- replay on/off.

The decisive experiment is not whether retrieval alone works, but whether **support formation + centroid recall + copy decision** jointly yield exact occurrence recovery.

---

## 7. Conclusion

We have presented a new canonical formulation of GFST-HMB in which exact long-context memory is treated as **sparse-support centroid recall over exact-token LUT blocks**. The local GFST and MPUC recurrence provide a compact streaming state substrate, but exact recall is delegated to a sparse associative layer that writes tokens into block-local cells and reads them back through significance-gated multi-block centroid selection. Phase binding, self-field bias, and selective replay are not auxiliary decorations; they are part of the mechanism that turns coarse block retrieval into occurrence-specific exact memory.

The paper's central claim is therefore different from a minimal geometric-router story. A single block key may still be useful for coarse routing, but exact long-context recall depends on **sparse write support, low-coherence cell structure, occurrence binding, and sparse multi-head readout**. The resulting system remains linear in storage, avoids full hidden-state caching, and gives a concrete path toward exact long-context memory on practical hardware. The remaining scientific question is clear and testable: whether this sparse-support centroid memory can sustain occurrence-specific exact recall at scale.

---

## References

[1] Tri Dao and Albert Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*. arXiv:2405.21060, 2024.

[2] Cameron J. Blocker, Haroon Raja, Jeffrey A. Fessler, and Laura Balzano. *Dynamic Subspace Estimation with Grassmannian Geodesics*. arXiv:2303.14851, 2023.

[3] Andras Sasfi, Alberto Padoan, Ivan Markovsky, and Florian Dorfler. *GREAT: Grassmannian REcursive Algorithm for Tracking & Online System Identification*. arXiv:2412.09052, 2024.

[4] Lucie Fontaine and Frederic Alexandre. *Semantic and episodic memories in a predictive coding model of the neocortex*. arXiv:2509.01987, 2025.

[5] Daniel Szelogowski. *Engram Memory Encoding and Retrieval: A Neurocomputational Perspective*. arXiv:2506.01659, 2025.

[6] Sang-Yoon Kim and Woochang Lim. *Effect of Adult-Born Immature Granule Cells on Pattern Separation in The Hippocampal Dentate Gyrus*. arXiv:2301.05387, 2023.

[7] Reza Ahmadvand, Sara Safura Sharif, and Yaser Mike Banad. *A Unified Phase-native Computational Principle Governs Hippocampal Spike Timing and Neural Coding*. arXiv:2603.19690, 2026.

[8] Liyang Lu, Haochen Wu, Wenbo Xu, Zhaocheng Wang, and H. Vincent Poor. *Hierarchically Block-Sparse Recovery With Prior Support Information*. arXiv:2511.06173, 2025.

[9] Canyi Lu, Huan Li, and Zhouchen Lin. *Optimized Projections for Compressed Sensing via Direct Mutual Coherence Minimization*. arXiv:1508.03117, 2015.

[10] The Tien Mai. *Optimal sparse phase retrieval via a quasi-Bayesian approach*. arXiv:2504.09509, 2025.

[11] Jean-Luc Starck, Sandrine Pires, and Alexandre Refregier. *Weak Lensing Mass Reconstruction using Wavelets*. arXiv:astro-ph/0503373, 2005.

[12] Jean-Luc Starck, Jalal Fadili, Seth Digel, Bin Zhang, and James Chiang. *Source detection using a 3D sparse representation: application to the Fermi gamma-ray space telescope*. arXiv:0904.3299, 2009.

[13] J. Schmitt, J. L. Starck, J. M. Casandjian, J. Fadili, and I. Grenier. *Poisson Denoising on the Sphere: Application to the Fermi Gamma Ray Space Telescope*. arXiv:1003.5613, 2010.

[14] Gianni V. Vinci, Andrea Galluzzi, and Maurizio Mattia. *Beyond catastrophic forgetting in associative networks with self-interactions*. arXiv:2504.04560, 2025.

[15] Adria Labay-Mora, Roberta Zambrini, and Gian Luca Giorgi. *Quantum associative memory with a single driven-dissipative nonlinear oscillator*. arXiv:2205.09491, 2022.
