eval "$(conda shell.bash hook)"
conda activate p2
python MPCN/train.py --dataset $1 --gpu $2 --dmax 20 --epochs 25 --early_stop 3 \
	                 --hdim $3 --rnn_size $3 --use_cudnn 1
conda deactivate