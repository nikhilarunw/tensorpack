
import os
from .utils import logger

# issue#7378 may happen with custom opencv. It doesn't hurt to disable opencl
os.environ['OPENCV_OPENCL_RUNTIME'] = ''
try:
    # issue#1924 may happen on old systems
    import cv2  # noqa
except ImportError:
    pass
else:
    if int(cv2.__version__.split('.')[0]) == 3:
        cv2.ocl.setUseOpenCL(False)
    # check if cv is built with cuda
    info = cv2.getBuildInformation().split('\n')
    for line in info:
        if 'use cuda' in line.lower():
            answer = line.split()[-1].lower()
            if answer == 'yes':
                # issue#1197
                logger.warn("OpenCV is built with CUDA support. "
                            "This may cause slow initialization or sometimes segfault with TensorFlow.")
            break

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'  # issue#9339
os.environ['TF_AUTOTUNE_THRESHOLD'] = '3'   # use more warm-up
os.environ['TF_AVGPOOL_USE_CUDNN'] = '1'   # issue#8566

__version__ = '0.3.0'
