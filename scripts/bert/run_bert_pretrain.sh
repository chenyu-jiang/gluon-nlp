#!/bin/bash -e
export MXNET_GPU_WORKER_NTHREADS=1
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

export TRUNCATE_NORM="${TRUNCATE_NORM:-1}"
export GLUON_NLP_BRANCH="${GLUON_NLP_BRANCH:-bert-byteprofile}"
export DMLC_ROLE="${DMLC_ROLE:-worker}"


PY_VERSION="3"

if [ "$PY_VERSION" = "3" ]; then
	PYTHON="python3"
elif [ "$PY_VERSION" = "2" ]; then	
	PYTHON="python"
else
	echo "Python version error"
fi

BYTEPS_PATH=`${PYTHON} -c "import byteps as bps; path=str(bps.__path__); print(path.split(\"'\")[1])"`
echo "BYTEPS_PATH:${BYTEPS_PATH}" 
# BYTEPS_PATH: /usr/local/lib/python3.6/site-packages/byteps-0.1.0-py3.6-linux-x86_64.egg/byteps/torch

##----------------------------------- 		Modify MXNet 	  ----------------------------------- 
# \TODO: direct get the gradient names in bytePS without modifying MXNet python part
if [ $DMLC_ROLE = "worker" ]; then
	echo "Modify MXNet for workers"
	MX_PATH=`${PYTHON} -c "import mxnet; path=str(mxnet.__path__); print(path.split(\"'\")[1])"`
	echo "MX_PATH: $MX_PATH"
	${PYTHON} /root/byteps-sniff/launcher/insert_code.py \
			--target_file="$MX_PATH/module/executor_group.py" \
			--start="        self.arg_names = symbol.list_arguments()" \
			--end="        self.aux_names = symbol.list_auxiliary_states()" \
			--indent_level=2 \
			--content_str="import os
_param_names = [name for i, name in enumerate(self.arg_names) if name in self.param_names]
path = os.environ.get('BYTEPS_TRACE_DIR', '.') + '/' + os.environ.get('BYTEPS_RANK') + '_' + os.environ.get('BYTEPS_LOCAL_RANK') + '/'
if path:
	if not os.path.exists(path):
		os.makedirs(path)
	with open(os.path.join(path, 'arg_namesINpara_names.txt'), 'w') as f:
		for name in _param_names:
			f.write('%s\n' % name) # output execution graph"
else
	echo "No need to modify mxnet for server/scheduler."
fi

## To avoid integrating multiple operators into one single events
# \TODO: may influence the performance
export MXNET_EXEC_BULK_EXEC_TRAIN=0


# ---------------------- start to run ----------------------

DATA="/tmp/wiki_en_uncased_data/wiki_en_uncased_0*"
OPTIMIZER="bertadam"

# optimizer parameters
export LR=0.00354;   
export OPTIONS=--synthetic_data\ --eval_use_npz; 
export WARMUP_RATIO=0.1;          
export NUMSTEPS=30;   
export CKPTDIR=ckpt_stage1_lamb_16k-682a361-c5fd6fc-0412-cu90; 
export ACC=1;         
export GPUS=0

# start
export TRUNCATE_NORM="${TRUNCATE_NORM:-1}"
export LAMB_BULK="${LAMB_BULK:-30}"
export EPS_AFTER_SQRT="${EPS_AFTER_SQRT:-1}"
export NUMSTEPS="${NUMSTEPS:-900000}"
export DTYPE="${DTYPE:-float16}"
export ACC="${ACC:-1}"
export MODEL="${MODEL:-bert_24_1024_16}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-128}"
export MAX_PREDICTIONS_PER_SEQ="${MAX_PREDICTIONS_PER_SEQ:-20}"
export LR="${LR:-0.000625}"
export LOGINTERVAL="${LOGINTERVAL:-10}"
export CKPTDIR="${CKPTDIR:-ckpt_stage1_lamb}"
export CKPTINTERVAL="${CKPTINTERVAL:-300000000}"
export OPTIMIZER="${OPTIMIZER:-lamb}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.003125}"
export NVIDIA_VISIBLE_DEVICES="${GPUS:-0,1,2,3,4,5,6,7}"

export NCCL_MIN_NRINGS="${NCCL_MIN_NRINGS:-16}"
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD="${MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD:-120}"
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD="${MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD:-120}"
export MXNET_SAFE_ACCUMULATION="${MXNET_SAFE_ACCUMULATION:-1}"
export OPTIONS="${OPTIONS:- }"
export DATA="${DATA:-/data/book-corpus/book-corpus-large-split/*.train,/data/enwiki/enwiki-feb-doc-split/*.train}"
export DATAEVAL="${DATAEVAL:-/data/book-corpus/book-corpus-large-split/*.test,/data/enwiki/enwiki-feb-doc-split/*.test}"

echo "NVIDIA_VISIBLE_DEVICES: $NVIDIA_VISIBLE_DEVICES"
readarray -d , -t strarr <<<"$NVIDIA_VISIBLE_DEVICES"
TOTAL_BATCH_SIZE=$(($DMLC_NUM_WORKER*${#strarr[*]}*10))
echo "total batch size is $TOTAL_BATCH_SIZE"

python3 /root/byteps-sniff/launcher/launch.py \
	python3 /root/gluon-nlp/scripts/bert/run_pretraining.py \
		--data=$DATA \
		--data_eval=$DATAEVAL \
		--optimizer $OPTIMIZER \
		--warmup_ratio $WARMUP_RATIO \
		--num_steps $NUMSTEPS \
		--ckpt_interval $CKPTINTERVAL \
		--dtype $DTYPE \
		--ckpt_dir $CKPTDIR \
		--lr $LR \
		--accumulate $ACC \
		--model $MODEL \
		--max_seq_length $MAX_SEQ_LENGTH \
		--max_predictions_per_seq $MAX_PREDICTIONS_PER_SEQ \
		--num_data_workers 4 \
		--no_compute_acc \
		--log_interval $LOGINTERVAL \
		--total_batch_size $TOTAL_BATCH_SIZE \
		--total_batch_size_eval $TOTAL_BATCH_SIZE \
		--gpus $NVIDIA_VISIBLE_DEVICES \
		--synthetic_data \
		--comm_backend byteps