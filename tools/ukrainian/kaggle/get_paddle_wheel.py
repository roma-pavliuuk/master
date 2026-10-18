"""CPU-only Kaggle kernel: download the paddlepaddle-gpu wheel once, so GPU kernels can install it
from this kernel's output (attached as kernel source) instead of spending ~20 min of GPU quota on a
slow download from the Paddle mirror."""
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "download", "--no-deps", "-d", "/kaggle/working/wheels",
                       "paddlepaddle-gpu==3.2.0", "-i", "https://www.paddlepaddle.org.cn/packages/stable/cu126/"])
subprocess.call("ls -la /kaggle/working/wheels", shell=True)
