#!/bin/bash
# exit 0 iff the engine is idle (sessions 0, no gpu_job, no queued jobs)
h=$(curl -s -m 3 127.0.0.1:10051/health) || exit 2
python3 -c "import json,sys;d=json.loads(sys.argv[1]);sys.exit(0 if d['sessions']==0 and d['gpu_job'] is None and d['queued_jobs']==0 else 1)" "$h"
