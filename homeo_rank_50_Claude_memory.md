# Working Memory: homeo_rank_50

## Origin

This exploration starts from **iter_116** of the `simulation_oscillatory_rank_50` exploration. That run achieved good rate constant recovery with the following settings:
- lr_k=5E-3, lr_sub=5E-3, lr_node=1E-4, lr_embedding=1E-4
- coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_MLP_sub_diff=6, coeff_k_floor=1.0
- batch_size=8, data_augmentation_loop=5000 (claude uses 1000)
- MLP_sub: hidden_dim=64, n_layers=4; MLP_node: hidden_dim=32, n_layers=2

The new goal: jointly recover rate constants k, homeostatic slopes (MLP_node), and embedding separation.

## Knowledge from Previous Exploration

- lr_k=5E-3 and lr_sub=5E-3 work well for k recovery at this regime
- MLP_node was kept small (coeff_MLP_node_L1=1.0, lr_node=1E-4)
- Homeostatic signal is ~1000x weaker than reaction dynamics
- n_layers_node >= 2 required for embedding-dependent slopes (Tanh nonlinearity needed)

## Established Principles

(none yet -- to be populated from exploration results)

## Current Block (Block 1)

### Hypothesis
Can MLP_node learn correct homeostatic slopes while maintaining rate constant recovery, by tuning lr_node, lr_embedding, and coeff_MLP_node_L1?

### Iterations This Block
(none yet)

### Emerging Observations
(none yet)
