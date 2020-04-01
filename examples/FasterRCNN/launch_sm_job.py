# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from sagemaker import get_execution_role
import sagemaker as sage
from sagemaker.estimator import Estimator
from sagemaker.inputs import FileSystemInput
import datetime
import subprocess
import sys

def get_str(cmd):
    content = subprocess.check_output(cmd, shell=True)
    return str(content)[2:-3]

account = get_str("echo $(aws sts get-caller-identity --query Account --output text)")
region = 'ap-southeast-1' #get_str("echo $(aws configure get region)")
image = str(sys.argv[1])
sess = sage.Session()
image_name=f"{account}.dkr.ecr.{region}.amazonaws.com/{image}"
sagemaker_iam_role = str(sys.argv[2])
num_gpus = 1
num_nodes = 1
instance_type = 'ml.p2.xlarge'
custom_mpi_cmds = []

job_name = "maskrcnn-{}x{}-{}".format(num_nodes, num_gpus, image)

s3_bucket = "smart-invoice"
prefix = "tensorpack/data" 
s3train = f's3://{s3_bucket}/{prefix}/train'
train = sage.session.s3_input(s3train, distribution='FullyReplicated', 
                        content_type='application/tfrecord', s3_data_type='S3Prefix')

output_path = f's3://{s3_bucket}/{prefix}/sagemaker_training_release'


# lustre_input = FileSystemInput(file_system_id='fs-03f556d03c3c590a2',
#                                file_system_type='FSxLustre',
#                                directory_path='/fsx',
#                                file_system_access_mode='ro')





data_channels = {'train': train}

hyperparams = {"sagemaker_use_mpi": "False",
               "sagemaker_process_slots_per_host": num_gpus,
               "num_gpus":num_gpus,
               "num_nodes": num_nodes,
               "load": "COCO-MaskRCNN-R50FPN2x.npz",
               "logdir": "/opt/ml/output",
               "custom_mpi_cmds": custom_mpi_cmds,
               "config": 'DATA.BASEDIR=/opt/ml/input/data MODE_FPN=True "DATA.VAL=(\'balloon_val\',)"  "DATA.TRAIN=(\'balloon_train\',)" TRAIN.BASE_LR=1e-3 TRAIN.EVAL_PERIOD=0 "TRAIN.LR_SCHEDULE=[1000]" "PREPROC.TRAIN_SHORT_EDGE_SIZE=[600,1200]" TRAIN.CHECKPOINT_PERIOD=1 DATA.NUM_WORKERS=1'}

estimator = Estimator(image_name, role=sagemaker_iam_role, output_path=output_path,
                      train_instance_count=num_nodes,
                      train_instance_type=instance_type,
                      # train_instance_type='local_gpu',
                      # sagemaker_session=sess,
                      train_volume_size=125,
                      base_job_name=job_name,
                      #   subnets=['subnet-21ac2f2e'],
                      #   security_group_ids=['sg-a21b02eb'],
                      hyperparameters=hyperparams)

estimator.fit(inputs=data_channels, logs=True)