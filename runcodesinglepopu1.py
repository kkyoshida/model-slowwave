import subprocess
import os

outputpath = r'data' 
if not os.path.exists(outputpath):
    os.makedirs(outputpath)

outputpath = r'figure' 
if not os.path.exists(outputpath):
    os.makedirs(outputpath)

os.chdir('code')

subprocess.run('python slowwave.py', shell=True)
subprocess.run('python slowwaveach.py', shell=True)
subprocess.run('python slowwaveinhibitory.py', shell=True)

subprocess.run('python plotslowwaveclear.py', shell=True)
subprocess.run('python plotslowwaveclearach.py', shell=True)
subprocess.run('python plotslowwaveclearinhibitory.py', shell=True)


