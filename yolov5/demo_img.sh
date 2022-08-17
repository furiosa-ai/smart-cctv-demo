# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate py39torch
./demo_stop.sh
PYTHONPATH=. OMP_NUM_THREADS=4 \
#python run/mcmot_demo.py --cfg cfg/mot_fur15_pipeline_aicas_test_img.yaml --device furiosa "$@" 2> error.txt 1> output.txt
#python run/mcmot_demo.py --cfg cfg/mot_fur15_pipeline_aicas_test_img.yaml --device furiosa "$@"
python run/mcmot_demo.py --cfg cfg/mot_fur15_pipeline_aicas_test_img.yaml --device cpu "$@"
