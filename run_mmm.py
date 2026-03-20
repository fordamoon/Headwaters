import subprocess
subprocess.run(["jupyter","nbconvert","--to","notebook","--execute","--inplace","mmm_demo.ipynb"])
