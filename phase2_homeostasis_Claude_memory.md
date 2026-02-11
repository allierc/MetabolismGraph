# Phase 2 Working Memory: phase2_homeostasis

## Knowledge Base

### Time Step Comparison
| Block | ts=4 score | ts=16 score | ts=32 score | ts=64 score | Best strategy | Key finding |
| ----- | ---------- | ----------- | ----------- | ----------- | ------------- | ----------- |
| 1 (Batch 1) | pending | pending | pending | pending | baseline | Initial baseline comparison |

### Established Principles
- MLP_node starts at zero with near-zero gradients; standard LRs (1E-3) may be insufficient
- lr_node_homeo should be higher than lr_emb_homeo (MLP_node must learn first)
- Longer rollouts accumulate more homeostatic signal but have noisier gradients
- homeostatic signal is ~1000x weaker than reaction dynamics

### Open Questions
- Which time_step (4, 16, 32, 64) works best for Phase 2?
- Is lr_node_homeo=0.01 sufficient to escape zero initialization?
- How many iterations (data_augmentation_loop) needed for convergence?

---

## Previous Block Summary
(None - this is the first block)

---

## Current Block (Block 1)

### Block Info
- Block: 1, Batch: 1
- Iterations: 1-4 (first batch)
- Goal: Establish baseline comparison across time_steps

### Strategy Under Test
**Baseline Configuration** - Same parameters across all 4 slots to isolate the effect of time_step

### Batch 1 Configuration (Iterations 1-4)
| Slot | time_step | lr_node_homeo | lr_emb_homeo | data_aug_loop | batch_size |
| ---- | --------- | ------------- | ------------ | ------------- | ---------- |
| 0 | 4 | 0.01 | 0.001 | 1000 | 8 |
| 1 | 16 | 0.01 | 0.001 | 1000 | 8 |
| 2 | 32 | 0.01 | 0.001 | 1000 | 8 |
| 3 | 64 | 0.01 | 0.001 | 1000 | 8 |

### Rationale
- lr_node_homeo=0.01: 10x higher than default to escape zero initialization trap
- lr_emb_homeo=0.001: Lower than node LR since embeddings should follow MLP_node learning
- data_augmentation_loop=1000: Standard duration for initial test
- Identical params across slots to isolate time_step effect

### Iterations This Block

### Emerging Observations
(Will be updated after Batch 1 results)
