### this script is used to predict forest disturbance on a single FORCE tile
### it is used in the inference_on_all_force_tiles.py script to loop over all FORCE tiles
### and predict the forest disturbance for each tile

import torch
from torch.utils.data import DataLoader

from exp.exp_anomaly_detection import Exp_Anomaly_Detection


from pandas.tseries.offsets import DateOffset
from datetime import datetime
import argparse
import time



if __name__ == "__main__":


    ### create the parser
    parser = argparse.ArgumentParser()

    ### tile name
    parser.add_argument('--tile_name', type=str, required=True)

    ### path to the model (trained on LUX study site as spatial hold-out, meaning that it is most representative for Germany)
    parser.add_argument('--model_path', type=str, required=True)
    
    ### path to FORCE tile directories
    parser.add_argument('--force_tile_base_path', type=str, required=False, default='/force/FORCE/C1/L2/ard')

    ### where to save the results?
    parser.add_argument('--result_tile_base_path', type=str, required=True)

    ### forest mask base path containing the tile folders and the forest mask.tif
    ### this will ensure that each tile does contain forest (as it is pre-filtered)
    parser.add_argument('--forest_mask_base_path', type=str, required=True)

    ## end date of time series
    parser.add_argument('--end_date', type=str, required=False)

    ## batch size
    parser.add_argument('--batch_size', type=int, required=False, default=2048)

    ### model parameters from training
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--task_name', type=str, default="anomaly_detection")
    parser.add_argument('--model_id', type=str, default="RsTs")
    parser.add_argument('--model', type=str, default="TimesNet_RsTs")
    parser.add_argument('--fixed_norm', type=int, default=1)
    parser.add_argument('--data', type=str, default="RsTs")
    parser.add_argument('--info', type=str, default="dc_fxnrm_nk8_bs256_lr00001_qai_shfflyrs")
    parser.add_argument('--d_ff', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--num_kernels', type=int, default=8)
    parser.add_argument('--embed', type=str, default="learnedsincos")
    parser.add_argument('--dropout', type=float, default=0.5)  # not used in inference, but must be defined
    parser.add_argument('--top_k', type=int, default=1)  # not used in inference, but must be defined
    parser.add_argument('--seq_len', type=int, default=200)  # 200 timesteps
    parser.add_argument('--num_workers', type=int, default=70)  # 12 == max on eolab platform
    parser.add_argument('--band_list', nargs='+', 
                        help='choose from (BLUE, GREEN, RED, REDEDGE1, REDEDGE2, REDEDGE3, BROADNIR, NIR, SWIR1, SWIR2)', 
                        default=["NIR", "SWIR1", "SWIR2"])
    ### despite python 0-indexing; the last three bands out of 10 are 8, 9 and 10 (NIR, SWIR1 and SWIR2)
    ### the logic in numpy is a bit different!!!
    ### fixed normalization parameter: mean
    # coniferous CRSWIR best seed: 0.716687169266206 # all forest types CRSWIR: 0.763984164508625
    parser.add_argument('--overall_mean', nargs='+', default=[0.716687169266206]) 
    ### fixed normalization parameter: standard deviation
    # coniferous CRSWIR best seed: 0.0821973190461412 # all forest types CRSWIR: 0.0721499625416236
    parser.add_argument('--overall_std', nargs='+', default=[0.0821973190461412]) 

    ### add argument use_gpu
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    ### further model parameters (mostly not relevant, only preventing code crashes)
    ### this can be polished later, once the code is running
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size', required=False)
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size', required=False)
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    ### add 0 or 1 question if indices should be used or not
    parser.add_argument('--use_indices', type=int, default=0, help='use vegetation indices: 0 = no, 1 = yes')
    ### optionally, parse a list of indices or bands to be used
    parser.add_argument('--indices_bands', nargs='+', help='List of indices to be used as input data', 
                        default=['CRSWIR'])

    ### parse the argument
    args = parser.parse_args()

    start_time = time.perf_counter()

    # print('CUDA available: ')
    # print(torch.cuda.is_available())

    print("Current time (beginning of script):")
    print(datetime.now())

    ### initialize anomaly detection task
    Exp = Exp_Anomaly_Detection
    exp = Exp(args)  # set experiments
    print('>>>>>>>start testing script<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.predict()
    torch.cuda.empty_cache()