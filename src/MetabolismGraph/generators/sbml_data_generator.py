"""Generate MetabolismGraph training data from an SBML kinetic model.

Extracts stoichiometry + boundary species from SBML, then simulates with
PDE_MichaelisMenten + time-varying external stimuli (external sources).

Without time-varying boundary input the system equilibrates to steady state
(activity rank ~1). The external stimuli are analogous to sensory stimuli in
connectome-gnn — they drive the network and produce rich dynamics.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
import torch

from MetabolismGraph.utils import to_numpy, get_equidistant_points
from MetabolismGraph.generators.utils import (
    plot_stoichiometric_matrix,
    plot_stoichiometric_eigenvalues,
    plot_metabolism_concentrations,
)


# ── SBML XML parser ─────────────────────────────────────────────────────────

def _parse_sbml(sbml_path):
    """Parse SBML Level 2 file → species list + reaction list."""
    tree = ET.parse(sbml_path)
    root = tree.getroot()
    tag = root.tag
    ns_uri = tag[tag.find('{') + 1:tag.find('}')] if '{' in tag else ''
    ns = {'sbml': ns_uri} if ns_uri else {}

    def find(parent, path):
        return parent.findall(path, ns) if ns else parent.findall(path.replace('sbml:', ''))

    species_info = []
    for s in find(root, './/sbml:listOfSpecies/sbml:species'):
        species_info.append({
            'id': s.get('id'),
            'initial_concentration': float(s.get('initialConcentration', '0')),
            'boundary': s.get('boundaryCondition', 'false').lower() == 'true',
        })

    reaction_info = []
    for rxn in find(root, './/sbml:listOfReactions/sbml:reaction'):
        subs = [(r.get('species'), float(r.get('stoichiometry', '1')))
                for r in find(rxn, 'sbml:listOfReactants/sbml:speciesReference')]
        prods = [(p.get('species'), float(p.get('stoichiometry', '1')))
                 for p in find(rxn, 'sbml:listOfProducts/sbml:speciesReference')]
        reaction_info.append({'id': rxn.get('id'), 'substrates': subs, 'products': prods})

    return species_info, reaction_info


def _build_stoichiometry_with_boundary(species_info, reaction_info, device=None):
    """Build S matrix + edge lists, separating floating vs boundary species."""
    floating = [s for s in species_info if not s['boundary']]
    boundary_all = [s for s in species_info if s['boundary']]
    var_id_to_idx = {s['id']: i for i, s in enumerate(floating)}

    # Find boundary species that appear in reactions (not just enzymes)
    reaction_species = set()
    for rxn in reaction_info:
        for sp, _ in rxn['substrates'] + rxn['products']:
            reaction_species.add(sp)
    stimulus_species = [s for s in boundary_all if s['id'] in reaction_species]
    stim_id_to_idx = {s['id']: i for i, s in enumerate(stimulus_species)}

    species_names = [s['id'] for s in floating]
    reaction_names = [r['id'] for r in reaction_info]
    n_met, n_rxn = len(floating), len(reaction_info)

    S_np = np.zeros((n_met, n_rxn), dtype=np.float32)
    sub_edges, all_edges, stim_sub_edges = [], [], []

    for j, rxn in enumerate(reaction_info):
        for sp, stoich in rxn['substrates']:
            if sp in var_id_to_idx:
                i = var_id_to_idx[sp]
                S_np[i, j] -= stoich
                sub_edges.append((i, j, float(stoich)))
                all_edges.append((i, j, -float(stoich)))
            elif sp in stim_id_to_idx:
                stim_sub_edges.append((stim_id_to_idx[sp], j, float(stoich)))
        for sp, stoich in rxn['products']:
            if sp in var_id_to_idx:
                i = var_id_to_idx[sp]
                S_np[i, j] += stoich
                all_edges.append((i, j, float(stoich)))

    stoich_graph = {
        'sub': (torch.tensor([e[0] for e in sub_edges], dtype=torch.long, device=device),
                torch.tensor([e[1] for e in sub_edges], dtype=torch.long, device=device),
                torch.tensor([e[2] for e in sub_edges], dtype=torch.float32, device=device)),
        'all': (torch.tensor([e[0] for e in all_edges], dtype=torch.long, device=device),
                torch.tensor([e[1] for e in all_edges], dtype=torch.long, device=device),
                torch.tensor([e[2] for e in all_edges], dtype=torch.float32, device=device)),
    }
    if stim_sub_edges:
        stoich_graph['stimulus_sub'] = (
            torch.tensor([e[0] for e in stim_sub_edges], dtype=torch.long, device=device),
            torch.tensor([e[1] for e in stim_sub_edges], dtype=torch.long, device=device),
            torch.tensor([e[2] for e in stim_sub_edges], dtype=torch.float32, device=device),
        )

    S_tensor = torch.tensor(S_np, dtype=torch.float32, device=device)
    return S_tensor, stoich_graph, species_names, reaction_names, floating, stimulus_species


# ── Stimulus generation ─────────────────────────────────────────────────────

def _generate_stimulus(stimulus_info, n_frames, seed=42):
    """Generate time-varying boundary species concentrations.

    Each boundary species oscillates around its baseline with temporally
    correlated noise, mimicking fluctuating nutrient supply, cofactor
    recycling, and metabolic demand.

    Returns
    -------
    stimulus : ndarray (n_frames+1, n_boundary)
        Time-varying concentrations for each boundary species.
    """
    rng = np.random.RandomState(seed)
    n_stim = len(stimulus_info)
    baselines = np.array([s['initial_concentration'] for s in stimulus_info], dtype=np.float64)

    stimulus = np.zeros((n_frames + 1, n_stim), dtype=np.float32)

    for i in range(n_stim):
        c0 = baselines[i]
        if c0 < 1e-6:
            # near-zero boundary species: keep at zero
            stimulus[:, i] = c0
            continue

        # Multi-scale temporally correlated signal: sum of slow + medium + fast
        # Very slow (tau ~ n_frames/3) creates large sustained concentration swings
        # Medium and fast components add structure
        # Total amplitude ~1-3x baseline so the metabolic network is strongly driven
        n_components = 3
        taus = [rng.uniform(n_frames * 0.2, n_frames * 0.5),
                rng.uniform(n_frames * 0.05, n_frames * 0.15),
                rng.uniform(n_frames * 0.01, n_frames * 0.04)]
        amps = [rng.uniform(0.8, 1.5), rng.uniform(0.3, 0.7), rng.uniform(0.1, 0.3)]

        signal = np.zeros(n_frames + 1, dtype=np.float64)
        for k in range(n_components):
            filt_len = int(taus[k] * 3)
            filt = np.exp(-np.arange(filt_len) / taus[k])
            filt /= filt.sum()
            noise = rng.randn(n_frames + 1 + filt_len)
            comp = np.convolve(noise, filt, mode='full')[:n_frames + 1]
            comp = (comp - comp.mean()) / (comp.std() + 1e-8)
            signal += amps[k] * comp

        # Clamp to positive, allow swings from 0.05x to 4x baseline
        stimulus[:, i] = np.clip(c0 * (1.0 + signal), c0 * 0.05, c0 * 4.0).astype(np.float32)

    return stimulus


# ── Euler integration with stimulus ─────────────────────────────────────────

def _simulate_michaelis_menten(config, stoich_graph, n_metabolites, initial_conc,
                                stimulus, n_frames, delta_t, device):
    """Simulate PDE_MichaelisMenten with time-varying external stimuli."""
    from torch_geometric import data as pyg_data
    from MetabolismGraph.generators.PDE_MichaelisMenten import PDE_MichaelisMenten
    from tqdm import trange

    model = PDE_MichaelisMenten(config=config, stoich_graph=stoich_graph, device=device)
    model.to(device)

    xc, yc = get_equidistant_points(n_points=n_metabolites)
    pos = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2

    x = torch.zeros((n_metabolites, 8), dtype=torch.float32, device=device)
    x[:, 0] = torch.arange(n_metabolites, dtype=torch.float32, device=device)
    x[:, 1:3] = pos.clone().detach()
    x[:, 3] = torch.tensor(initial_conc, dtype=torch.float32, device=device)

    # Convert stimulus to tensor
    stim_tensor = torch.tensor(stimulus, dtype=torch.float32, device=device) \
        if stimulus is not None else None

    x_list = []
    y_list = []

    for it in trange(n_frames + 1, ncols=100, desc='Euler'):
        bnd_stim = stim_tensor[it] if stim_tensor is not None else None

        with torch.no_grad():
            dataset = pyg_data.Data(x=x, pos=x[:, 1:3])
            y = model(dataset, dt=delta_t, stimulus=bnd_stim)

        x_list.append(to_numpy(x.clone()))
        y_list.append(to_numpy(y.clone()))

        du = y.squeeze()
        x[:, 3] = x[:, 3] + du * delta_t
        x[:, 3] = torch.clamp(x[:, 3], min=0.0)

    x_list = np.array(x_list)
    y_list = np.array(y_list)

    return model, x_list, y_list


# ── Main entry point ────────────────────────────────────────────────────────

def sbml_data_generate(config, visualize=True, device=None, bSave=True):
    """Generate training data from an SBML kinetic model."""
    sim = config.simulation
    dataset_name = config.dataset
    sbml_path = sim.sbml_file

    folder = f'./graphs_data/{dataset_name}/'
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'{folder}/Fig/', exist_ok=True)

    if not os.path.isabs(sbml_path):
        sbml_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', sbml_path)
        sbml_abs = os.path.abspath(sbml_abs)
    else:
        sbml_abs = sbml_path

    t_end = sim.sbml_t_end
    n_frames = sim.n_frames
    delta_t = t_end / n_frames

    # ── 1. Parse SBML ────────────────────────────────────────────────────

    print(f"Loading SBML: {sbml_abs}")
    species_info, reaction_info = _parse_sbml(sbml_abs)
    S_tensor, stoich_graph, species_names, reaction_names, floating_info, stimulus_info = \
        _build_stoichiometry_with_boundary(species_info, reaction_info, device=device)

    n_metabolites = len(species_names)
    n_reactions = len(reaction_names)
    sim.n_metabolites = n_metabolites
    sim.n_reactions = n_reactions

    if stimulus_info:
        sim.stimulus_concentrations = [s['initial_concentration'] for s in stimulus_info]
        sim.stimulus_names = [s['id'] for s in stimulus_info]

    S_np = to_numpy(S_tensor)
    initial_conc = np.array([s['initial_concentration'] for s in floating_info], dtype=np.float64)

    print(f"Floating species: {n_metabolites}")
    print(f"Reactions: {n_reactions}")
    print(f"S matrix: {S_np.shape}, non-zero: {np.count_nonzero(S_np)}")
    if stimulus_info:
        bnd_strs = [f"{s['id']}={s['initial_concentration']:.2f}" for s in stimulus_info]
        print(f"External sources: {len(stimulus_info)} ({', '.join(bnd_strs)})")

    # ── 2. Generate external stimuli ─────────────────────────────────────

    stimulus = None
    if stimulus_info:
        print(f"Generating time-varying external stimuli ({len(stimulus_info)} species)...")
        stimulus = _generate_stimulus(stimulus_info, n_frames, seed=sim.seed)
        print(f"  Stimulus shape: {stimulus.shape}")
        for i, s in enumerate(stimulus_info):
            stim = stimulus[:, i]
            print(f"  {s['id']:10s}: baseline={s['initial_concentration']:.2f}, "
                  f"range=[{stim.min():.2f}, {stim.max():.2f}]")

    # ── 3. Simulate ──────────────────────────────────────────────────────

    print(f"Simulating: t=[0, {t_end}], {n_frames} frames, dt={delta_t:.6f}")
    print("  Backend: PDE_MichaelisMenten + Euler (with external stimuli)")

    model, x_list, y_list = _simulate_michaelis_menten(
        config, stoich_graph, n_metabolites, initial_conc,
        stimulus, n_frames, delta_t, device)

    conc_matrix = x_list[:, :, 3]
    print(f"Concentration range: [{conc_matrix.min():.4f}, {conc_matrix.max():.4f}]")
    print(f"x_list: {x_list.shape}, y_list: {y_list.shape}")
    print(f"dc/dt range: [{y_list.min():.4f}, {y_list.max():.4f}]")

    # ── 4. Save ──────────────────────────────────────────────────────────

    if bSave:
        torch.save(S_tensor, f'{folder}/stoichiometry.pt')
        torch.save(stoich_graph, f'{folder}/stoich_graph.pt')
        np.save(f'{folder}/x_list_0.npy', x_list)
        np.save(f'{folder}/y_list_0.npy', y_list)
        if stimulus is not None:
            np.save(f'{folder}/stimulus.npy', stimulus)
        torch.save(model.state_dict(), f'{folder}/gt_model.pt')

        metadata = {
            'source': f'SBML: {sim.sbml_file}',
            'n_metabolites': n_metabolites,
            'n_reactions': n_reactions,
            'delta_t': delta_t,
            'n_frames': n_frames,
            't_end': t_end,
            'species_names': species_names,
            'reaction_names': reaction_names,
            'initial_concentrations': conc_matrix[0].tolist(),
            'boundary_species': [{'id': s['id'], 'concentration': s['initial_concentration']}
                                 for s in stimulus_info] if stimulus_info else [],
        }
        torch.save(metadata, f'{folder}/metadata.pt')
        print(f"Saved to {folder}")

    # ── 5. Plots ─────────────────────────────────────────────────────────

    plot_stoichiometric_matrix(S_tensor, dataset_name)
    plot_stoichiometric_eigenvalues(S_tensor, dataset_name)

    from MetabolismGraph.models.utils import analyze_data_svd
    svd_results = analyze_data_svd(x_list, folder, config=config, save_in_subfolder=False)
    activity_rank = svd_results.get('activity', {}).get('rank_99', None)

    plot_metabolism_concentrations(x_list, n_metabolites, n_frames, dataset_name, delta_t, activity_rank=activity_rank)

    bnd_names = [s['id'] for s in stimulus_info] if stimulus_info else None

    print("plotting kinograph ...")
    _plot_kinograph_labeled(conc_matrix, species_names, delta_t, folder,
                            stimulus, bnd_names)

    print("plotting traces ...")
    _plot_metabolite_traces(conc_matrix, species_names, delta_t, folder,
                            stimulus, bnd_names)
    _plot_dcdt_traces(y_list, species_names, delta_t, folder)

    if stimulus is not None:
        print("plotting external stimuli ...")
        _plot_stimulus(stimulus, bnd_names, delta_t, folder)

    # ── Summary ──────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"SBML Data Generation Complete")
    print(f"  Source: {sim.sbml_file}")
    print(f"  Metabolites: {n_metabolites} ({', '.join(species_names)})")
    print(f"  Reactions: {n_reactions}")
    if stimulus_info:
        print(f"  External sources: {len(stimulus_info)} ({', '.join(s['id'] for s in stimulus_info)})")
    print(f"  Frames: {n_frames}, dt={delta_t:.6f}, t_end={t_end}")
    if activity_rank is not None:
        print(f"  Activity rank (99%): {activity_rank}")
    log_k = model.log_k.detach().cpu()
    print(f"  Vmax range: [{10**log_k.min().item():.4f}, {10**log_k.max().item():.4f}]")
    log_km = model.log_km.detach().cpu()
    print(f"  Km range: [{10**log_km.min().item():.4f}, {10**log_km.max().item():.4f}]")
    print(f"  Output: {folder}")
    print(f"{'='*60}")


# ── Plotting ────────────────────────────────────────────────────────────────

WARMUP_TIME = 10.0  # SBML units to skip (initial transient)


def _cut_warmup(data, delta_t):
    """Return data with initial transient removed, and the new time offset."""
    skip = int(WARMUP_TIME / delta_t)
    skip = min(skip, data.shape[0] - 10)  # keep at least 10 frames
    return data[skip:], skip * delta_t


def _plot_kinograph_labeled(conc_matrix, species_names, delta_t, folder,
                            stimulus=None, stimulus_names=None):
    """Kinograph with metabolite names + stimulus strip at bottom (transient cut)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    conc, t0 = _cut_warmup(conc_matrix, delta_t)
    stim = None
    if stimulus is not None:
        stim, _ = _cut_warmup(stimulus, delta_t)

    n_frames, n_met = conc.shape
    n_ticks = 6
    tick_pos = np.linspace(0, n_frames - 1, n_ticks, dtype=int)
    tick_labels = [f'{t0 + p * delta_t:.0f}' for p in tick_pos]

    if stim is not None and stimulus_names is not None:
        n_stim = stim.shape[1]
        stim_norm = np.zeros_like(stim)
        for i in range(n_stim):
            s = stim[:, i]
            smin, smax = s.min(), s.max()
            if smax - smin > 1e-10:
                stim_norm[:, i] = (s - smin) / (smax - smin)
        cmax = np.abs(conc).max()
        stim_display = stim_norm * cmax
        combined = np.vstack([stim_display.T, conc.T])
        all_names = stimulus_names + species_names
        n_total = n_stim + n_met
    else:
        combined = conc.T
        all_names = species_names
        n_total = n_met
        n_stim = 0

    fig, ax = plt.subplots(1, 1, figsize=(14, max(4, n_total * 0.3)))
    im = ax.imshow(combined, aspect='auto', cmap='viridis', origin='lower', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('concentration (mM)', fontsize=13)
    cbar.ax.tick_params(labelsize=9)

    ax.set_yticks(range(n_total))
    ylabels = [f'[{name}]' if i < n_stim else name for i, name in enumerate(all_names)]
    ax.set_yticklabels(ylabels, fontsize=9)
    if n_stim > 0:
        ax.axhline(n_stim - 0.5, color='red', linewidth=1.5, linestyle='--', alpha=0.8)

    ax.set_xlabel('Time (SBML units)', fontsize=13)
    ax.set_ylabel('Metabolite', fontsize=13)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.set_title(f'Kinograph — {n_met} metabolites + {n_stim} external sources (t>{WARMUP_TIME:.0f})')
    plt.tight_layout()
    plt.savefig(f'{folder}/kinograph.png', dpi=150)
    plt.close()


def _plot_metabolite_traces(conc_matrix, species_names, delta_t, folder,
                            stimulus=None, stimulus_names=None):
    """Stacked traces, normalized by post-transient amplitude, transient cut."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    conc, t0 = _cut_warmup(conc_matrix, delta_t)
    stim = None
    if stimulus is not None:
        stim, _ = _cut_warmup(stimulus, delta_t)

    n_frames, n_met = conc.shape
    t_axis = t0 + np.arange(n_frames) * delta_t
    n_stim = stim.shape[1] if stim is not None else 0
    n_total = n_stim + n_met
    step = 1.2

    fig, ax = plt.subplots(1, 1, figsize=(14, max(6, n_total * 0.4)))

    # Plot stimulus traces (red, at bottom) — normalized by post-transient range
    if stim is not None and stimulus_names is not None:
        for i in range(n_stim):
            trace = stim[:, i]
            tmin, tmax = trace.min(), trace.max()
            if tmax - tmin > 1e-10:
                trace_norm = (trace - tmin) / (tmax - tmin)
            else:
                trace_norm = np.zeros_like(trace)
            ax.plot(t_axis, trace_norm + i * step, linewidth=1.2, alpha=0.7, color='red')

    # Plot metabolite traces (blue, above) — normalized by post-transient range
    for i in range(n_met):
        trace = conc[:, i]
        tmin, tmax = trace.min(), trace.max()
        if tmax - tmin > 1e-10:
            trace_norm = (trace - tmin) / (tmax - tmin)
        else:
            trace_norm = np.zeros_like(trace)
        ax.plot(t_axis, trace_norm + (n_stim + i) * step, linewidth=1.2, alpha=0.8, color='#1f77b4')

    all_names = [f'[{n}]' for n in (stimulus_names or [])] + species_names
    ax.set_yticks([i * step + 0.5 for i in range(n_total)])
    ax.set_yticklabels(all_names, fontsize=9)
    if n_stim > 0:
        ax.axhline(n_stim * step - step / 2, color='red', linewidth=1, linestyle='--', alpha=0.5)

    ax.set_xlabel('Time (SBML units)', fontsize=13)
    ax.set_ylabel('Species', fontsize=13)
    ax.set_title(f'Traces — {n_met} metabolites + {n_stim} external sources (t>{WARMUP_TIME:.0f})', fontsize=13)
    ax.set_xlim([t_axis[0], t_axis[-1]])
    ax.set_ylim([-0.5, n_total * step + 0.5])
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(f'{folder}/traces.png', dpi=150)
    plt.close()

    # Print post-transient amplitude for each metabolite
    print(f"  Post-transient amplitude (t>{WARMUP_TIME:.0f}):")
    for i in range(n_met):
        amp = conc[:, i].max() - conc[:, i].min()
        mean = conc[:, i].mean()
        print(f"    {species_names[i]:10s}: amplitude={amp:.6f}, mean={mean:.4f}, relative={amp/(mean+1e-10):.4f}")


def _plot_dcdt_traces(y_list, species_names, delta_t, folder):
    """Stacked dc/dt traces (transient cut)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    y_cut, t0 = _cut_warmup(y_list, delta_t)
    n_frames, n_met = y_cut.shape[0], y_cut.shape[1]
    t_axis = t0 + np.arange(n_frames) * delta_t
    dcdt = y_cut[:, :, 0]

    fig, ax = plt.subplots(1, 1, figsize=(14, max(6, n_met * 0.4)))
    step = 1.2
    for i in range(n_met):
        trace = dcdt[:, i]
        tmin, tmax = trace.min(), trace.max()
        if tmax - tmin > 1e-10:
            trace_norm = (trace - tmin) / (tmax - tmin)
        else:
            trace_norm = np.zeros_like(trace)
        ax.plot(t_axis, trace_norm + i * step, linewidth=1.2, alpha=0.8)

    ax.set_yticks([i * step + 0.5 for i in range(n_met)])
    ax.set_yticklabels(species_names, fontsize=9)
    ax.set_xlabel('Time (SBML units)', fontsize=13)
    ax.set_title(f'dc/dt Traces — {n_met} metabolites (t>{WARMUP_TIME:.0f})', fontsize=13)
    ax.set_xlim([t_axis[0], t_axis[-1]])
    ax.set_ylim([-0.5, n_met * step + 0.5])
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(f'{folder}/traces_dcdt.png', dpi=150)
    plt.close()


def _plot_stimulus(stimulus, stimulus_names, delta_t, folder):
    """Dedicated stimulus plot (transient cut)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    stim, t0 = _cut_warmup(stimulus, delta_t)
    n_frames, n_stim = stim.shape
    t_axis = t0 + np.arange(n_frames) * delta_t

    fig, ax = plt.subplots(1, 1, figsize=(14, max(4, n_stim * 0.4)))
    step = 1.2
    for i in range(n_stim):
        trace = stim[:, i]
        tmin, tmax = trace.min(), trace.max()
        if tmax - tmin > 1e-10:
            trace_norm = (trace - tmin) / (tmax - tmin)
        else:
            trace_norm = np.zeros_like(trace)
        ax.plot(t_axis, trace_norm + i * step, linewidth=1.2, alpha=0.8, color='red')

    ax.set_yticks([i * step + 0.5 for i in range(n_stim)])
    ax.set_yticklabels(stimulus_names, fontsize=9)
    ax.set_xlabel('Time (SBML units)', fontsize=13)
    ax.set_title(f'External Sources — {n_stim} species (t>{WARMUP_TIME:.0f})', fontsize=13)
    ax.set_xlim([t_axis[0], t_axis[-1]])
    ax.set_ylim([-0.5, n_stim * step + 0.5])
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(f'{folder}/stimulus.png', dpi=150)
    plt.close()
