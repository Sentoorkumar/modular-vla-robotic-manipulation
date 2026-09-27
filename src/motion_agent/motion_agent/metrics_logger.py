#!/usr/bin/env python3
"""
metrics_logger.py
=================
Tiny, standalone, REMOVABLE metrics logger for the thesis pipeline.

Design goals:
  - Minimal: each node adds just 1-2 lines.
  - Non-invasive: if this file is missing or logging is OFF, the calls
    become harmless no-ops. Nodes keep working exactly as before.
  - Simple output: one CSV file, one row per event, easy to analyze.

How a node uses it:
    from motion_agent.metrics_logger import log_event
    log_event('llm', 'llm_decision_done', trial_id=cmd_id, target='can 1')

Turn logging ON/OFF with an environment variable (no code change):
    export THESIS_METRICS=1      # ON
    unset THESIS_METRICS         # OFF (calls become no-ops)

Output file (default): ~/thesis_metrics/run_log.csv
Override with:         export THESIS_METRICS_FILE=/path/to/file.csv

CSV columns:
    wall_time_iso, unix_time, node, event, trial_id, target, pose, extra
"""

import os
import time
import csv
import threading

# ------------------------------------------------------------------
# Configuration (read once at import)
# ------------------------------------------------------------------
_ENABLED = os.environ.get('THESIS_METRICS', '0') == '1'

_DEFAULT_DIR = os.path.expanduser('~/thesis_metrics')
_LOG_FILE = os.environ.get(
    'THESIS_METRICS_FILE',
    os.path.join(_DEFAULT_DIR, 'run_log.csv')
)

_HEADER = ['wall_time_iso', 'unix_time', 'node', 'event',
           'trial_id', 'target', 'pose', 'extra']

_lock = threading.Lock()
_initialized = False


def _ensure_file():
    """Create the log dir + CSV header once, safely."""
    global _initialized
    if _initialized:
        return
    try:
        os.makedirs(os.path.dirname(_LOG_FILE), exist_ok=True)
        if not os.path.exists(_LOG_FILE) or os.path.getsize(_LOG_FILE) == 0:
            with open(_LOG_FILE, 'w', newline='') as f:
                csv.writer(f).writerow(_HEADER)
        _initialized = True
    except Exception:
        # Never let logging break the node.
        pass


def log_event(node, event, trial_id='', target='', pose='', extra=''):
    """
    Append one event row to the CSV. Safe no-op if logging is OFF or
    anything goes wrong. NEVER raises.

    Args:
        node:     which node ('stt', 'llm', 'perception', 'motion')
        event:    short event name ('llm_decision_done', 'grasp_close', ...)
        trial_id: an id to group events of one command (optional)
        target:   object label, e.g. 'can 1' (optional)
        pose:     'standing' / 'lying' (optional)
        extra:    any extra note or number as string (optional)
    """
    if not _ENABLED:
        return
    try:
        _ensure_file()
        row = [
            time.strftime('%Y-%m-%dT%H:%M:%S'),
            f'{time.time():.6f}',
            str(node), str(event), str(trial_id),
            str(target), str(pose), str(extra),
        ]
        with _lock:
            with open(_LOG_FILE, 'a', newline='') as f:
                csv.writer(f).writerow(row)
    except Exception:
        # Swallow everything — logging must never crash the pipeline.
        pass


def is_enabled():
    """True if metrics logging is currently ON."""
    return _ENABLED


def log_file_path():
    """Where the log is being written."""
    return _LOG_FILE
