FROM deeplearning
RUN pip install git+https://github.com/tensorpack/tensorpack.git
RUN pip install Cython numpy==1.18.2 opencv-python tensorflow-gpu==1.13.2
RUN for i in cython 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' scipy; do pip install $i; done
COPY ./dataset /opt/ml/code/dataset
COPY ./modeling /opt/ml/code/modeling
COPY ./utils /opt/ml/code/utils
COPY ./convert_d2 /opt/ml/code/convert_d2
COPY ./*.py /opt/ml/code/
# RUN mkdir -p /opt/ml/input/data
# ADD ./data /opt/ml/input/data
# RUN mkdir -p /opt/ml/model
# ADD ./COCO-MaskRCNN-R50FPN2x.npz /opt/ml/model/COCO-MaskRCNN-R50FPN2x.npz
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/:/usr/local/cuda/lib/:/usr/local/cuda-10.0/compat/
ENV SAGEMAKER_PROGRAM train.py
CMD python /opt/ml/code/train.py --config DATA.BASEDIR=/opt/ml/input/data MODE_FPN=True \
	"DATA.VAL=('balloon_val',)"  "DATA.TRAIN=('balloon_train',)" \
	TRAIN.BASE_LR=1e-3 TRAIN.EVAL_PERIOD=0 "TRAIN.LR_SCHEDULE=[1000]" \
	"PREPROC.TRAIN_SHORT_EDGE_SIZE=[600,1200]" TRAIN.CHECKPOINT_PERIOD=1 DATA.NUM_WORKERS=1 \
	--load COCO-MaskRCNN-R50FPN2x.npz --logdir train_log/balloon