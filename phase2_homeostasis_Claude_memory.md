# Phase 2 Working Memory: phase2_homeostasis

## Knowledge Base

### Time Step Comparison
| Block | ts=4 score | ts=16 score | ts=32 score | ts=64 score | Best strategy | Key finding |
| ----- | ---------- | ----------- | ----------- | ----------- | ------------- | ----------- |

### Established Principles
- MLP_node starts at zero with near-zero gradients; standard LRs (1E-3) may be insufficient
- lr_node_homeo should be higher than lr_emb_homeo because MLP_node must move first before embeddings can separate
- Longer rollouts (time_step) accumulate more homeostatic signal but have noisier gradients
- Homeostatic signal is ~1000x weaker than reaction dynamics

### Open Questions
- Which rollout length (4, 16, 32, 64) works best for Phase 2?
- Is lr_node_homeo=0.01 sufficient to escape zero initialization?
- How many training iterations (data_augmentation_loop) are needed?

---

## Previous Block Summary
(none - this is the first batch)

---

## Current Block (Block 1)

### Block Info
- Block 1, Batch 1 (iterations 1-4)
- Strategy: Baseline comparison across all time_steps

### Strategy Under Test
Hyperparameter-only baseline to isolate effect of rollout length (time_step). All 4 slots use identical training parameters:
- lr_node_homeo = 0.01 (higher than default to escape zero init)
- lr_emb_homeo = 0.001
- data_augmentation_loop = 1000
- batch_size = 8

### Initial Config (Batch 1)
| Slot | time_step | lr_node_homeo | lr_emb_homeo | data_aug_loop | batch_size |
|------|-----------|---------------|--------------|---------------|------------|
| 0    | 4         | 0.01          | 0.001        | 1000          | 8          |
| 1    | 16        | 0.01          | 0.001        | 1000          | 8          |
| 2    | 32        | 0.01          | 0.001        | 1000          | 8          |
| 3    | 64        | 0.01          | 0.001        | 1000          | 8          |

### Rationale
Following instruction guidelines:
1. First batch uses SAME params across all slots to establish baseline comparison of time_steps
2. lr_node_homeo=0.01 is 10x higher than default (0.001) per the instruction's "Key insight" that LRs of 1E-2 to 1E-1 are worth exploring
3. lr_emb_homeo=0.001 is lower than lr_node_homeo because MLP_node must learn first; embeddings follow
4. data_augmentation_loop=1000 as starting point; may increase if slopes move but haven't converged

### Iterations This Block

### Emerging Observations
(pending results from batch 1)
