"""Run the served q3 wrapper (~/llm/ds41/serve/ds41_serve.py) unchanged, plus og_resume.

The wrapper's source is executed as __main__ with one line added right before it
starts the omlx CLI (all its memory limits, guards and warm-up are in place by
then): og_resume.install(), which only acts on requests carrying x-ds41-resume.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WRAPPER = Path.home()/'llm/ds41/serve/ds41_serve.py'
ANCHOR = 'from omlx.cli import main\n'

source = WRAPPER.read_text()
if source.count(ANCHOR) != 1:
    raise SystemExit(f'{WRAPPER}: resume hook anchor not found exactly once')
source = source.replace(ANCHOR, 'import og_resume; og_resume.install()\n' + ANCHOR)
sys.path.insert(0, str(HERE))
sys.argv[0] = str(WRAPPER)
exec(compile(source, str(WRAPPER), 'exec'), {'__name__': '__main__', '__file__': str(WRAPPER)})
