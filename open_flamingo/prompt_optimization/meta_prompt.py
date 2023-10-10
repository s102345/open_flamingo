import subprocess
result = subprocess.run(['python', './eval/scorer.py'], capture_output=True)
print(result.stdout)