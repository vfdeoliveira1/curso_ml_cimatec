import sys

from reconhecimento_facial.cli import app

# O DeepFace escreve emojis no log; o console do Windows (cp1252) não consegue exibi-los
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", errors="replace")

app(prog_name="reconhecimento_facial")
