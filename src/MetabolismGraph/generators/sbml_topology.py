"""
Library-free loader: parse an FBA / constraint-based SBML model's STOICHIOMETRY
(no kinetics needed, no cobra/libsbml) into the generator's stoich_graph format,
so we can impose synthetic Michaelis--Menten kinetics on a REAL network topology
and test S-given k-recovery (the PDF's "realistic topology, S given" rung).

We parse only <reaction> elements and their reactant/product speciesReferences via
xml.etree (namespace-stripped). Internal species become metabolites; species flagged
boundaryCondition="true" become external stimulus sources (the stimulus_sub edges,
as in the glycolysis import).

Returns: stoich_graph dict {'sub','all','stimulus_sub'}, dense S (n_met x n_rxn),
species_names, reaction_names.
"""
import re
import xml.etree.ElementTree as ET
import numpy as np
import torch


def _tag(e):
    return e.tag.split('}')[-1]   # strip namespace


def _find(parent, name):
    return [c for c in parent if _tag(c) == name]


EXCHANGE_PAT = re.compile(r"(^R_EX_|^EX_|biomass|BIOMASS|^R_ATPM|ATPM|_exchange|sink_|SK_|DM_|demand)", re.I)


def _build(species, boundary, reactions, cols, keep_met, keep_rxn, device):
    """Re-index a kept metabolite/reaction subset into a stoich_graph + dense S."""
    keep_met = sorted(keep_met); keep_rxn = sorted(keep_rxn)
    mmap = {m: i for i, m in enumerate(keep_met)}
    rmap = {r: j for j, r in enumerate(keep_rxn)}
    S = np.zeros((len(keep_met), len(keep_rxn)), dtype=np.float32)
    sub_e, all_e, stim_e = [], [], []
    for j, col in cols:
        if j not in rmap:
            continue
        for sid, c in col.items():
            if c == 0.0 or sp_idx_global[sid] not in mmap:
                continue
            i = mmap[sp_idx_global[sid]]; jj = rmap[j]
            S[i, jj] = c; all_e.append((i, jj, c))
            if c < 0:
                (stim_e if boundary.get(sid) else sub_e).append((i, jj, abs(c)))

    def tri(edges):
        if not edges:
            z = torch.zeros(0, dtype=torch.long, device=device)
            return (z, z, torch.zeros(0, dtype=torch.float32, device=device))
        return (torch.tensor([e[0] for e in edges], dtype=torch.long, device=device),
                torch.tensor([e[1] for e in edges], dtype=torch.long, device=device),
                torch.tensor([e[2] for e in edges], dtype=torch.float32, device=device))
    sg = {"sub": tri(sub_e), "all": tri(all_e),
          "stimulus_sub": tri(stim_e) if stim_e else None}   # None -> model skips boundary branch
    return sg, torch.tensor(S, device=device), [species[m] for m in keep_met], [reactions[r] for r in keep_rxn]


def parse_fba_sbml(path, device="cpu", max_order=4, min_metabolites=2, compartment=None):
    """Parse stoichiometry from an FBA SBML, keeping only GENUINE enzymatic reactions:
    drop exchange/biomass/ATPM/sink pseudo-reactions, reactions with substrate order
    > max_order (lumped/biomass), and reactions with < min_metabolites species.
    Optionally restrict to a single compartment suffix (e.g. 'c' for cytosol).
    Returns (stoich_graph, S, species, reactions)."""
    global sp_idx_global
    root = ET.parse(path).getroot()
    model = next(c for c in root if _tag(c) == "model")

    species, boundary, compart = [], {}, {}
    for lo in _find(model, "listOfSpecies"):
        for sp in _find(lo, "species"):
            sid = sp.get("id")
            species.append(sid)
            boundary[sid] = (sp.get("boundaryCondition", "false") == "true")
            compart[sid] = sp.get("compartment", "")
    sp_index = {s: i for i, s in enumerate(species)}
    sp_idx_global = sp_index

    def refs(reaction, kind):
        out = []
        for lo in _find(reaction, kind):
            for sr in _find(lo, "speciesReference"):
                st = sr.get("stoichiometry", "1")
                try:
                    st = float(st)
                except ValueError:
                    st = 1.0
                out.append((sr.get("species"), st))
        return out

    reactions = []
    cols = []   # list of (rxn_idx, {species: signed_coeff})
    for lo in _find(model, "listOfReactions"):
        for rx in _find(lo, "reaction"):
            reactions.append(rx.get("id"))
            j = len(reactions) - 1
            col = {}
            for sid, st in refs(rx, "listOfReactants"):
                col[sid] = col.get(sid, 0.0) - st        # reactants negative
            for sid, st in refs(rx, "listOfProducts"):
                col[sid] = col.get(sid, 0.0) + st        # products positive
            cols.append((j, col))

    # ---- keep only genuine enzymatic reactions ----
    keep_rxn, keep_met = set(), set()
    for j, col in cols:
        rid = reactions[j]
        if EXCHANGE_PAT.search(rid):
            continue                                       # exchange/biomass/ATPM/sink
        order = sum(-c for c in col.values() if c < 0)     # substrate order |s|
        if order > max_order or order < 1:
            continue                                       # lumped/biomass / no substrate
        if len([c for c in col.values() if c != 0]) < min_metabolites:
            continue                                       # exchange / single-species
        if compartment is not None and not all(
                compart.get(sid, "").rstrip("_").endswith(compartment) or
                compart.get(sid, "") == compartment for sid in col):
            continue                                       # restrict to one compartment
        keep_rxn.add(j)
        keep_met.update(sp_index[sid] for sid in col if col[sid] != 0)
    return _build(species, boundary, reactions, cols, keep_met, keep_rxn, device)


def _S_to_stoich_graph(S, device):
    """Dense S (n_met x n_rxn) -> stoich_graph {sub, all, stimulus_sub(empty)}."""
    sub_e, all_e = [], []
    nz = np.argwhere(S != 0)
    for i, j in nz:
        c = float(S[i, j]); all_e.append((int(i), int(j), c))
        if c < 0:
            sub_e.append((int(i), int(j), abs(c)))

    def tri(edges):
        if not edges:
            z = torch.zeros(0, dtype=torch.long, device=device)
            return (z, z, torch.zeros(0, dtype=torch.float32, device=device))
        return (torch.tensor([e[0] for e in edges], dtype=torch.long, device=device),
                torch.tensor([e[1] for e in edges], dtype=torch.long, device=device),
                torch.tensor([e[2] for e in edges], dtype=torch.float32, device=device))
    return {"sub": tri(sub_e), "all": tri(all_e), "stimulus_sub": None}


def central_carbon_subgraph(path, compartment="c", max_reactions=120, device="cpu"):
    """For huge genome-scale models (yeast-GEM): first filter to genuine enzymatic
    cytosolic reactions (|s|<=4, no biomass/exchange), then grow a CONNECTED reaction
    set by BFS from the highest-degree metabolite until ~max_reactions reactions."""
    _, S, species, reactions = parse_fba_sbml(
        path, device="cpu", max_order=4, min_metabolites=2, compartment=compartment)
    S = S.numpy()
    if S.shape[1] <= max_reactions:
        keep_rxn = list(range(S.shape[1]))
    else:
        # BFS from the highest-degree metabolite, adding reactions that touch the frontier
        met_deg = (S != 0).sum(1)
        seed_met = int(np.argmax(met_deg))
        keep_rxn, seen_met, frontier = [], {seed_met}, [seed_met]
        while frontier and len(keep_rxn) < max_reactions:
            nxt = []
            for m in frontier:
                for j in np.nonzero(S[m] != 0)[0]:
                    if j in keep_rxn:
                        continue
                    keep_rxn.append(int(j))
                    if len(keep_rxn) >= max_reactions:
                        break
                    for m2 in np.nonzero(S[:, j] != 0)[0]:
                        if m2 not in seen_met:
                            seen_met.add(int(m2)); nxt.append(int(m2))
                if len(keep_rxn) >= max_reactions:
                    break
            frontier = nxt
    keep_rxn = sorted(set(keep_rxn))
    keep_met = sorted({int(i) for j in keep_rxn for i in np.nonzero(S[:, j] != 0)[0]})
    Ssub = S[np.ix_(keep_met, keep_rxn)]
    sg = _S_to_stoich_graph(Ssub, device)
    return sg, torch.tensor(Ssub, device=device), [species[i] for i in keep_met], [reactions[j] for j in keep_rxn]


if __name__ == "__main__":
    import sys
    p = sys.argv[1]
    if "yeast-GEM" in p:
        sg, S, sp, rx = central_carbon_subgraph(p, max_reactions=120)
    else:
        sg, S, sp, rx = parse_fba_sbml(p)
    print(f"{p}: S={tuple(S.shape)}  sub_edges={sg['sub'][0].numel()} "
          f"all_edges={sg['all'][0].numel()} stim_edges={sg['stimulus_sub'][0].numel()}")
    neg = torch.clamp(-S, min=0)
    print("substrate order |s| per reaction:",
          {int(k): int(v) for k, v in zip(*torch.unique(neg.sum(0).round().int(), return_counts=True))})
