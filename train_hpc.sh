#PBS -lselect=1:ncpus=4:mem=256gb:ngpus=1:gpu_type=RTX6000
#PBS -lwalltime=24:0:0

cd ~/AudioCLIP

rm -r mlruns/

IP_HEAD=`hostname -I | tr ' ' '\n' | grep -m1 ^10.`
echo $IP_HEAD

singularity exec \
                --bind /rds/general/user/ss7412/projects/box-migration-ss7412/live/Model_training_da$
                --nv \
                --env IP_HEAD=$IP_HEAD \
                audioclip_latest.sif \
                bash -c "ray start --head \
                --temp-dir /rds/general/user/ss7412/home/AudioCLIP/ \
                --node-ip-address $IP_HEAD \
                --port=6379 \
                && python -u lightning_trainer/train_pipeline.py \
                --grid_search False \
                --config config_hpc.yaml"
