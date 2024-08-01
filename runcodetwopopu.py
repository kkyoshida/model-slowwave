import subprocess
import os

outputpath = r'data' 
if not os.path.exists(outputpath):
    os.makedirs(outputpath)

outputpath = r'figure' 
if not os.path.exists(outputpath):
    os.makedirs(outputpath)

os.chdir('code')

subprocess.run('python twopopu.py', shell=True)
subprocess.run('python twopopubidir1.py', shell=True)
subprocess.run('python twopopubidir2.py', shell=True)
subprocess.run('python plottwopopuclear.py', shell=True)
subprocess.run('python plottwopopuclearbidir1.py', shell=True)
subprocess.run('python plottwopopuclearbidir2.py', shell=True)

