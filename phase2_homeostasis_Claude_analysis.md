# Phase 2 Experiment Log: phase2_homeostasis (parallel)

## Block 1, Batch 1: Baseline Initialization

### Strategy
Baseline comparison to isolate the effect of rollout length (time_step). All 4 slots use identical training parameters.

### Configuration
| Slot | time_step | lr_node_homeo | lr_emb_homeo | data_aug_loop | batch_size |
|------|-----------|---------------|--------------|---------------|------------|
| 0    | 4         | 0.01          | 0.001        | 1000          | 8          |
| 1    | 16        | 0.01          | 0.001        | 1000          | 8          |
| 2    | 32        | 0.01          | 0.001        | 1000          | 8          |
| 3    | 64        | 0.01          | 0.001        | 1000          | 8          |

### Rationale
- lr_node_homeo=0.01 is 10x higher than default (0.001) to help MLP_node escape zero initialization
- lr_emb_homeo=0.001 is lower because MLP_node must learn first before embeddings can separate
- Same params across all slots isolates the effect of time_step (rollout length)
- Longer rollouts accumulate more homeostatic signal but have noisier gradients

### Key Questions for This Batch
1. Which rollout length (4, 16, 32, 64) produces the best phase2_score?
2. Does lr_node_homeo=0.01 help MLP_node escape zero initialization (phase2_node_magnitude > 0)?
3. Do any slopes move toward GT values (-0.001, -0.002)?

---

