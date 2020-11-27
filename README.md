Code for testing MRDMN



run:

python test.py --batch-size 64 --num-class 200 --resolution 224 -- val-data-dir data/birds-ori-test.rec --weights_path weights/MRDM_birds-0000.params --num_gpus 1
