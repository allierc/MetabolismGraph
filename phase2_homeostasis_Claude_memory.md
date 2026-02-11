# Phase 2 Working Memory: phase2_homeostasis

## Fresh Start — Block 1

### Context
Previous exploration (13 blocks, 104 iterations) used a **supervised contrastive loss** that read ground-truth metabolite type labels (`x[:, 6]`) during training. This constituted label leakage: embeddings separated because they were told the correct types, not because MLP_node discovered type-dependent regulation. The supervised contrastive loss has been removed from the code.

### What was learned (retaining legitimate findings only)

**Signal characteristics:**
- Homeostatic signal is ~1000x weaker than reaction dynamics
- MLP_node starts at zero with near-zero gradients; standard LRs (1E-3) insufficient
- lr_node_homeo must be higher than lr_emb_homeo (MLP_node must learn first)

**Time step behavior (WITHOUT contrastive loss — from Blocks 1-2 before contrastive was added):**
- time_step=4: too short for homeostatic signal to accumulate
- time_step=16: marginal signal
- time_step=32: best balance of signal accumulation vs gradient noise (Block 1: 0.38)
- time_step=64: strongest signal but noisier gradients, highly stochastic

**Training strategies (legitimate):**
- Signal amplification (10x): makes homeostatic gradient comparable to reaction gradient
- Offset penalty: suppresses constant-output solutions, forces slope learning
- Gradient accumulation (4x): reduces variance from single-rollout BPTT
- Gradient clipping: stabilizes long-rollout BPTT
- Kaiming re-initialization of hidden layers: enables gradient flow through Tanh

**Key open problem:**
- Without supervised contrastive loss, embedding separation depends entirely on MLP_node producing type-differentiated outputs. If MLP_node learns a single average slope for all metabolites, there is no gradient signal to push embeddings apart. This chicken-and-egg problem is the core challenge.

### Established Principles
- DO NOT use ground-truth labels (`x[:, 6]`) during training — label leakage
- Embeddings must self-organize from learned MLP_node behavioral differences only
- The previous "best results" (80-92% avg_slope_ratio) were achieved with supervised contrastive loss and are not valid baselines

### Block 1 Plan
- Start with legitimate strategies only: amplification + offset penalty + gradient accumulation + gradient clipping
- Sweep lr_node_homeo across slots at different time_steps
- Establish unsupervised baselines before attempting any auxiliary losses
