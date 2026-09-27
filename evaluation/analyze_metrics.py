#!/usr/bin/env python3
"""
analyze_metrics.py
==================
Reads the thesis metrics CSV (written by the ROS2 nodes + Isaac judge)
and prints thesis-ready tables:

  1. Latency per stage (LLM decision time, motion sequence time, end-to-end)
  2. Motion success (standing vs lying) from sequence_complete_ok
  3. Sort outcomes from the Isaac judge (in correct bin? lifted? knocked off?)

Run it AFTER your measurement runs:
    python3 analyze_metrics.py                 # uses ~/thesis_metrics/run_log.csv
    python3 analyze_metrics.py /path/to.csv    # custom path

Nothing here touches the robot — it only reads the log file.
"""

import csv
import os
import sys
from collections import defaultdict


def load(path):
    rows = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def fnum(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def parse_extra(extra):
    """Parse 'k=v;k=v' into a dict."""
    d = {}
    for part in str(extra).split(';'):
        if '=' in part:
            k, v = part.split('=', 1)
            d[k.strip()] = v.strip()
    return d


# ------------------------------------------------------------------
def latency_report(rows):
    """
    Pair events per trial to compute latencies.
    Uses unix_time. Groups by trial_id (the object label / user text).
    """
    print("\n" + "=" * 62)
    print(" LATENCY REPORT (seconds)")
    print("=" * 62)

    # collect timestamps per (trial_id, event)
    ev = defaultdict(dict)   # trial_id -> event -> unix_time
    for r in rows:
        t = fnum(r['unix_time'])
        if t is None:
            continue
        ev[r['trial_id']].setdefault(r['event'], t)

    llm_lat, motion_lat, e2e_lat = [], [], []
    for tid, e in ev.items():
        # LLM: voice_received -> command_published
        if 'voice_received' in e and 'command_published' in e:
            llm_lat.append(e['command_published'] - e['voice_received'])
        # Motion: sequence_start -> sequence_end
        if 'sequence_start' in e and 'sequence_end' in e:
            motion_lat.append(e['sequence_end'] - e['sequence_start'])
        # End-to-end: voice_received -> sequence_end (if both exist)
        if 'voice_received' in e and 'sequence_end' in e:
            e2e_lat.append(e['sequence_end'] - e['voice_received'])
        # Motion-only end-to-end: command_received -> sequence_end
        elif 'command_received' in e and 'sequence_end' in e:
            e2e_lat.append(e['sequence_end'] - e['command_received'])

    def stat(name, vals):
        if not vals:
            print(f"  {name:28s}: (no data)")
            return
        avg = sum(vals) / len(vals)
        print(f"  {name:28s}: n={len(vals):3d}  "
              f"avg={avg:6.2f}  min={min(vals):6.2f}  max={max(vals):6.2f}")

    stat("LLM decision", llm_lat)
    stat("Motion sequence", motion_lat)
    stat("End-to-end", e2e_lat)


# ------------------------------------------------------------------
def motion_success_report(rows):
    """
    Success from motion node logs.
    sequence_start (with pose) = attempt.
    sequence_complete_ok       = full success (all 9/10 steps done).
    """
    print("\n" + "=" * 62)
    print(" MOTION SEQUENCE SUCCESS (from motion node)")
    print("=" * 62)

    attempts = defaultdict(int)   # pose -> count
    completes = defaultdict(int)  # pose -> count

    # attempts: count sequence_start by pose
    for r in rows:
        if r['node'] == 'motion' and r['event'] == 'sequence_start':
            attempts[r['pose'] or 'unknown'] += 1
        if r['node'] == 'motion' and r['event'] == 'sequence_complete_ok':
            completes[r['pose'] or 'unknown'] += 1

    print(f"  {'Pose':12s} {'Attempts':>9s} {'Completed':>10s} {'Rate':>8s}")
    print("  " + "-" * 42)
    total_a = total_c = 0
    for pose in sorted(set(list(attempts) + list(completes))):
        a = attempts.get(pose, 0)
        c = completes.get(pose, 0)
        total_a += a
        total_c += c
        rate = (100.0 * c / a) if a else 0.0
        print(f"  {pose:12s} {a:9d} {c:10d} {rate:7.1f}%")
    print("  " + "-" * 42)
    rate = (100.0 * total_c / total_a) if total_a else 0.0
    print(f"  {'TOTAL':12s} {total_a:9d} {total_c:10d} {rate:7.1f}%")


# ------------------------------------------------------------------
def sort_outcome_report(rows):
    """
    Outcomes from the Isaac judge (objective, position-based).
    """
    print("\n" + "=" * 62)
    print(" SORT OUTCOMES (from Isaac position judge)")
    print("=" * 62)

    outcomes = defaultdict(int)
    lifted_count = 0
    total = 0
    # A "trial_result" row is logged per object per trial. Only objects that
    # actually changed are interesting, but we count all and bucket outcomes.
    for r in rows:
        if r['node'] == 'isaac' and r['event'] == 'trial_result':
            d = parse_extra(r['extra'])
            outcome = d.get('outcome', 'unknown')
            outcomes[outcome] += 1
            if d.get('lifted') == '1':
                lifted_count += 1
            total += 1

    if total == 0:
        print("  (no Isaac judge data)")
        return

    print(f"  {'Outcome':16s} {'Count':>7s} {'Share':>8s}")
    print("  " + "-" * 34)
    for k in sorted(outcomes):
        c = outcomes[k]
        print(f"  {k:16s} {c:7d} {100.0*c/total:7.1f}%")
    print("  " + "-" * 34)
    print(f"  Objects lifted at least once: {lifted_count}/{total} "
          f"({100.0*lifted_count/total:.1f}%)")

    # Interpret the useful success numbers
    sorted_ok = outcomes.get('sorted_correct', 0)
    print(f"\n  Interpretation:")
    print(f"    Sorted into CORRECT bin:        {sorted_ok}")
    print(f"    Sorted into WRONG bin:          {outcomes.get('sorted_wrong_bin', 0)}")
    print(f"    Knocked to floor (grasp fail):  {outcomes.get('on_floor', 0)}")
    print(f"    Lifted but not binned:          {outcomes.get('lifted_only', 0)}")


# ------------------------------------------------------------------
def per_object_report(rows):
    """
    Per-object breakdown keyed on the Isaac STAGE-TREE NAME.
    The judge logs each object's full stage path in the 'target' column,
    e.g. /World/NaturalBostonRoundBottle_A01_PR_NVD_01. We use the last
    part of that path as the object name.

    For each object we count only trials where it actually participated
    (was lifted, binned, or knocked off) so untouched objects sitting in
    the scene do not inflate the attempt count.

    This is the table that backs the 'big-diameter bottle fails to attach
    in the lying position' claim with real numbers.
    """
    print("\n" + "=" * 80)
    print(" PER-OBJECT BREAKDOWN (Isaac stage-tree name)")
    print("=" * 80)

    stats = {}

    def _get(name):
        if name not in stats:
            stats[name] = {'attempts': 0, 'lifted': 0, 'sorted': 0,
                           'floor': 0, 'on_table': 0}
        return stats[name]

    for r in rows:
        if r['node'] != 'isaac' or r['event'] != 'trial_result':
            continue
        d = parse_extra(r['extra'])
        outcome = d.get('outcome', 'unknown')
        lifted = d.get('lifted') == '1'
        name = str(r['target']).split('/')[-1]   # stage-tree name

        # Count only objects that actually took part in the trial.
        participated = lifted or outcome in (
            'sorted_correct', 'sorted_wrong_bin', 'on_floor',
            'lifted_only', 'moved_on_table')
        if not participated:
            continue

        s = _get(name)
        s['attempts'] += 1
        if lifted:
            s['lifted'] += 1
        if outcome in ('sorted_correct',):
            s['sorted'] += 1
        if outcome == 'on_floor':
            s['floor'] += 1
        if outcome in ('on_table', 'moved_on_table'):
            s['on_table'] += 1

    if not stats:
        print("  (no Isaac judge data yet)")
        return

    print(f"  {'Object (stage name)':44s} {'Att':>4s} {'Lift':>5s} "
          f"{'Sort':>5s} {'Floor':>6s} {'GraspRate':>10s} {'SortRate':>9s}")
    print("  " + "-" * 88)
    for name in sorted(stats):
        s = stats[name]
        a = s['attempts']
        grasp_rate = (100.0 * s['lifted'] / a) if a else 0.0
        sort_rate = (100.0 * s['sorted'] / a) if a else 0.0
        short = name if len(name) <= 44 else name[:41] + '...'
        print(f"  {short:44s} {a:4d} {s['lifted']:5d} {s['sorted']:5d} "
              f"{s['floor']:6d} {grasp_rate:9.1f}% {sort_rate:8.1f}%")
    print("  " + "-" * 88)
    print("  Att=attempts  Lift=lifted (grasp ok)  Sort=landed in a bin  "
          "Floor=knocked off")
    print("  A low GraspRate flags the hard cases — e.g. the large-diameter")
    print("  bottle that resists attaching when lying down.")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.expanduser('~/thesis_metrics/run_log.csv')
    if not os.path.exists(path):
        print(f"Log file not found: {path}")
        print("Run some measured trials first (with THESIS_METRICS=1).")
        return
    rows = load(path)
    print(f"Loaded {len(rows)} events from {path}")
    latency_report(rows)
    motion_success_report(rows)
    sort_outcome_report(rows)
    per_object_report(rows)
    print()


if __name__ == '__main__':
    main()
