"""External 50 ms RSS/footprint and lease watchdog; PID + process UUID bound."""
import ctypes
import json
import os
from pathlib import Path
import signal
import sys
import time

pid=int(sys.argv[1]); path=Path(sys.argv[2]); limit=float(sys.argv[3]) if len(sys.argv)>3 else 840; lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib')
buf=ctypes.create_string_buffer(512); start=time.monotonic(); identity=None; peak_rss=peak_fp=0; saved=0
while lib.proc_pid_rusage(pid,2,buf)==0:
    raw=buf.raw
    if identity is None:identity=raw[:16]
    if identity != raw[:16]:break
    peak_rss=max(peak_rss,int.from_bytes(raw[64:72],'little'))
    peak_fp=max(peak_fp,int.from_bytes(raw[72:80],'little'))
    elapsed=time.monotonic()-start
    stop=max(peak_fp,peak_rss)>245*2**30 or (limit>0 and elapsed>limit)
    if elapsed-saved>1 or stop:
        path.write_text(json.dumps(dict(pid=pid,elapsed=elapsed,peak_rss_gib=peak_rss/2**30,peak_footprint_gib=peak_fp/2**30,stop=stop)))
        saved=elapsed
    if stop:
        os.kill(pid,signal.SIGKILL)
        break
    time.sleep(.05)
