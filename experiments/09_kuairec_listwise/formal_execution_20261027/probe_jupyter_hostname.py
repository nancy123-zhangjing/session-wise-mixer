"""Single temporary CPU terminal probe for authorized Stage09 recovery."""
import contextlib
import datetime
import io
import json
import logging
from pathlib import Path
import re
from inspire import InspireClient

report = {"schema": "stage09_jupyter_hostname_probe_v1",
          "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "notebook": "nancy-stage09-prep-20261027-r1",
          "command": "hostname", "transport": "jupyter",
          "creates_new_workload": False, "requests_gpu": False}
logging.disable(logging.CRITICAL)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    try:
        with InspireClient(account="sii-shared", allow_browser=False,
                           timeout=15, operation_timeout=50) as client:
            result = client.notebooks.exec(report["notebook"],
                workspace="CPU资源空间", command="hostname", transport="jupyter",
                timeout=35, capture=True, max_output_bytes=4096)
        value = result.stdout.strip()
        report.update(returncode=result.returncode, completed=result.completed,
                      truncated=result.truncated,
                      hostname=value if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,252}", value) else None)
        report["passed"] = result.completed and result.returncode == 0 and bool(report["hostname"])
    except Exception as error:
        report.update(passed=False, exception_type=type(error).__name__)
path = Path(__file__).with_name("JUPYTER_HOSTNAME_PROBE_20261028.json")
with path.open("x", encoding="utf-8") as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps(report, ensure_ascii=False))
raise SystemExit(0 if report["passed"] else 1)
