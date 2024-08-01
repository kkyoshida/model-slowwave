import subprocess
import os

outputpath = r'data' 
if not os.path.exists(outputpath):
    os.makedirs(outputpath)

outputpath = r'figure' 
if not os.path.exists(outputpath):
    os.makedirs(outputpath)

os.chdir('code')

subprocess.run('python plotfrequency.py', shell=True)
subprocess.run('python phaseplane.py', shell=True)
subprocess.run('python samplewave.py', shell=True)
subprocess.run('python animation.py', shell=True)

