#!/bin/bash
# POST python code (file or stdin) to the engine's localhost dev exec hook; prints the JSON reply.
curl -s -m ${OG_TIMEOUT:-600} -X POST --data-binary @"${1:--}" http://127.0.0.1:10059/exec; echo
