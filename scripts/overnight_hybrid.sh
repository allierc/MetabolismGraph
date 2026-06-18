#!/bin/bash
# Overnight hybrid structured-MM training matrix. Two GPU lanes run sequentially in
# parallel; each config is trained then evaluated (evolution + Vmax recovery + rollout),
# with a one-line result appended to docs/hybrid_overnight_results.md. Resumable: a config
# with log/<cfg>/OVERNIGHT_DONE is skipped, so re-launching after a crash continues.
cd /workspace/MetabolismGraph || exit 1
P=/workspace/.conda_envs/neural-graph-linux/bin/python
RESULTS=docs/hybrid_overnight_results.md
SUM=/tmp/overnight_summary.txt

# phase-A length per config (for the evolution figure shading)
declare -A NA=(
  [ecoli_core_hybrid_oracle]=40 [ecoli_core_hybrid_joint]=0  [ecoli_core_hybrid]=15
  [ecoli_core_hybrid_fastramp]=8 [ecoli_core_hybrid_slowramp]=25
  [glyco_hybrid_oracle]=40 [glyco_hybrid]=15 [yeast_hybrid_oracle]=40 [yeast_hybrid]=15
)

run_one () {   # $1=config  $2=gpu
  local cfg="$1" gpu="$2"
  if [ -f "log/$cfg/OVERNIGHT_DONE" ]; then echo "[skip] $cfg (done)"; return; fi
  echo "[$(date +%H:%M)] START $cfg on cuda:$gpu"
  sed -i "s/^  device: .*/  device: cuda:$gpu/" "config/$cfg.yaml"
  $P GNN_Main.py -o train "$cfg" > "/tmp/ov_${cfg}.log" 2>&1
  # ---- evaluate ----
  $P figures/hybrid_evolution.py "$cfg" "${NA[$cfg]:-15}" > "/tmp/ov_eval_${cfg}.log" 2>&1
  $P figures/k_recovery.py "$cfg"     >> "/tmp/ov_eval_${cfg}.log" 2>&1
  $P figures/toy_dashboard.py "$cfg"  >> "/tmp/ov_eval_${cfg}.log" 2>&1
  # ---- scrape numbers ----
  local evo kr roll
  evo=$(grep -aoE "final Vmax R2=[-0-9.]+, Km R2=[-0-9.]+.*" "/tmp/ov_eval_${cfg}.log" | tail -1)
  kr=$(grep -aoE "raw R2=[-0-9.]+ +trimmed R2=[-0-9.]+ +outliers=[0-9]+/[0-9]+ \([0-9.]+%\)" "/tmp/ov_eval_${cfg}.log" | tail -1)
  roll=$(grep -aoE "rollout Pearson per-met=[-0-9.]+ / pooled=[-0-9.]+" "/tmp/ov_eval_${cfg}.log" | tail -1)
  printf -- "- **%s** [%s] | %s | k_recovery: %s | %s\n" "$cfg" "$(date +%H:%M)" "$evo" "$kr" "$roll" >> "$RESULTS"
  printf -- "%s | %s | %s | %s\n" "$cfg" "$evo" "$kr" "$roll" >> "$SUM"
  touch "log/$cfg/OVERNIGHT_DONE"
  echo "[$(date +%H:%M)] DONE  $cfg"
}

lane () { for c in "$@"; do run_one "$c" "$LANE_GPU"; done }

echo "# Hybrid overnight results ($(date))" >> "$RESULTS"

# Lane 0 (cuda:0) and Lane 1 (cuda:1), parallel
( LANE_GPU=0; lane ecoli_core_hybrid_oracle ecoli_core_hybrid ecoli_core_hybrid_fastramp glyco_hybrid_oracle yeast_hybrid_oracle ) &
L0=$!
( LANE_GPU=1; lane ecoli_core_hybrid_joint ecoli_core_hybrid_slowramp glyco_hybrid yeast_hybrid ) &
L1=$!
wait $L0 $L1
echo "" >> "$RESULTS"
echo "ALL DONE $(date)" >> "$RESULTS"
echo "=== OVERNIGHT MATRIX COMPLETE $(date) ==="
