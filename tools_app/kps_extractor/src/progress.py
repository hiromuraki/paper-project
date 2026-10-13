"""Machine-readable progress for the web worker; ordinary CLI output stays readable."""
import json
import os
import time

_last_stage = None
_last_sent = 0.0


def report_progress(stage, current=None, total=None, unit="", message="", force=False):
    global _last_stage, _last_sent
    if os.environ.get("KPS_PROGRESS") != "1":
        return
    now = time.monotonic()
    if not force and stage == _last_stage and now - _last_sent < 0.2 and current != total:
        return
    _last_stage, _last_sent = stage, now
    # A preceding newline prevents stderr tqdm's carriage-return output from swallowing the record.
    print("\nKPS_PROGRESS:" + json.dumps(dict(stage=stage, current=current, total=total,
          unit=unit, message=message), ensure_ascii=False), flush=True)
