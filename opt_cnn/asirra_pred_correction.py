
import numpy as np
from threading import Thread
from aux_fct_correction import *
import gc, time, sys, glob

#Comment to access system wide install
#sys.path.insert(0,glob.glob('../../src/build/lib.*/')[-1])
import CIANNA as cnn

load_epoch = 0
if (len(sys.argv) > 1):
	load_epoch = int(sys.argv[1])

#Change image test mode in aux_fct to change network resolution in all functions
init_data_gen(test_mode=1)

cnn.init(in_dim=i_ar([image_size,image_size]), in_nb_ch=3, out_dim=nb_class, 
	bias=0.1, b_size=128, comp_meth='C_CUDA', dynamic_load=1, 
	mixed_precision="FP16C_FP32A", adv_size=30, inference_only=1)

#Compute on only half the validation set to reduce memory footprint
input_test, targets_test = create_val_batch()
cnn.create_dataset("TEST", nb_keep_val, input_test[:,:], targets_test[:,:])

del (input_test, targets_test)
gc.collect()

cnn.load("net_save/net0_s%04d.dat"%load_epoch, load_epoch, bin=1)

cnn.forward(repeat=1, no_error=1, saving=2, drop_mode="AVG_MODEL")

start = time.perf_counter()
cnn.forward(no_error=1, saving=2, drop_mode="AVG_MODEL")
end = time.perf_counter()

cnn.perf_eval()
cnn.print_arch_tex("./arch/", "arch", activation=1, dropout=1)

compute_time = (end-start)*1000 #in miliseconds
score_eval(load_epoch,compute_time, 586114)
		
