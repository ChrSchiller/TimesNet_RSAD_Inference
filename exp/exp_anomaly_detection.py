from exp.exp_basic import Exp_Basic
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
import os
import warnings
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from datetime import datetime

import rasterio
from utils.read_mask_force_rasters import read_mask_force_rasters
from data_provider.data_loader import RsTsLoader
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def predict(self):

        ### additional variables that need to be defined
        ### forest mask was created by R script create_forest_mask_by_force_tiles.R
        ### using Copernicus Land Monitoring Service forest mask and the FORCE datacube on www.eo-lab.org
        FOREST_MASK = os.path.join(self.args.forest_mask_base_path, self.args.tile_name, 'forest_mask.tif')
        RESULT_PATH = os.path.join(self.args.result_tile_base_path, self.args.tile_name)

        ### create RESULT_PATH
        if not os.path.exists(self.args.result_tile_base_path):
            os.makedirs(self.args.result_tile_base_path)
        if not os.path.exists(RESULT_PATH):
            os.mkdir(RESULT_PATH)

        ### filter force tile for: S2A/B and relevant timesteps
        files = os.listdir(os.path.join(self.args.force_tile_base_path, self.args.tile_name))
        boa_filenames = list(sorted(filter(lambda x: 'SEN2' in x and 'BOA' in x, files)))

        ### define end date either by user or by last observation in the FORCE tile
        if self.args.end_date:
            ### end date provided by user
            ENDDATE = self.args.end_date
        else: 
            ### last date of time series/scenes in FORCE tile directory
            ENDDATE = boa_filenames[len(boa_filenames)-1][:8]
        ### start date defined as: January 1st of end_date year - 3
        STARTDATE = pd.to_datetime(pd.to_datetime(ENDDATE, format='%Y%m%d') - DateOffset(years=3))  # 3 years
        ### change date (day and month) to January 1st in STARTDATE
        STARTDATE = STARTDATE.replace(day=1, month=1)
        ### if STARTDATE is earlier than 1 January 2016, set it to 1 January 2016
        if STARTDATE < pd.to_datetime('20160101', format='%Y%m%d'):
            STARTDATE = pd.to_datetime('20160101', format='%Y%m%d')
        print("Start date: ")
        print(STARTDATE)
        print("End date: ")
        print(ENDDATE)

        ### get dates
        dates = [pd.to_datetime(s[:8], format='%Y%m%d') for s in boa_filenames]

        ### check which dates are later than startdate (others can be discarded)
        dates = [t for t in dates if t >= STARTDATE]

        ### use this information to filter the list of S2 scenes
        ### discard files before start date
        ### we want as few rasters to load as possible
        boa_filenames = boa_filenames[-len(dates):]

        ### also discard all observations later than enddate
        dates = [t for t in dates if t <= pd.to_datetime(pd.to_datetime(ENDDATE, format='%Y%m%d'))]
        boa_filenames = boa_filenames[:len(dates)]

        ### read forest mask
        with rasterio.open(FOREST_MASK) as frst:
            forest_mask = np.squeeze(frst.read())
            meta = frst.meta.copy()  ### extract metadata

        ### stack all relevant rasters
        ### filter with forest mask while stacking
        ### use numpy arrays instead of rasters right away
        print('Stacking all FORCE tiles in study period...')
        print("Current time:")
        print(datetime.now())

        ### create numpy array to store all rasters
        h, w = forest_mask.shape
        ### number of features is only CRSWIR here (1)
        all_boas = np.empty((h, w, len(boa_filenames), len(self.args.indices_bands)), dtype=np.float32)

        ### read all the files and write to raster stack (= numpy array)
        for (i, f) in enumerate(boa_filenames):
            all_boas[:, :, i, :] = read_mask_force_rasters(
                os.path.join(self.args.force_tile_base_path, self.args.tile_name, f), 
                forest_mask, self.args.band_list, self.args.overall_mean, self.args.overall_std)

        ### reshape by flattening over height and width
        ### makes it easier to loop through dataset parallely
        all_boas = all_boas.reshape((-1, len(boa_filenames), len(self.args.indices_bands)))

        ### locations of all pixels that should be considered for prediction are exactly the non-NA
        ### values in forest_mask!
        forest_mask = forest_mask.reshape((-1)) ### flatten the forest mask
        iter_locations = np.where(forest_mask == 1)[0]

        ### initialize Dataset class, DataLoader and model instance
        print("Creating Dataloader...")
        ### get the Dataset class (preparing the single pixel's time series)
        test_data = RsTsLoader(iter_locations, all_boas, dates, self.args.seq_len, 
                               use_indices=0, info="")

        ### prepare the DataLoader
        test_loader = DataLoader(
            test_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False)

        print('Loading model...')
        self.model.load_state_dict(torch.load(self.args.model_path))

        ### also mkdir with results folder for test results
        if not os.path.exists(self.args.result_tile_base_path):
            os.makedirs(self.args.result_tile_base_path)
        if not os.path.exists(os.path.join(self.args.result_tile_base_path, self.args.tile_name)):
            os.makedirs(os.path.join(self.args.result_tile_base_path, self.args.tile_name))

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        print('Predicting...')
        print("Current time:")
        print(datetime.now())
        
        ### implementing the same loop as in original code
        ### but with batch_x_mark (doy) included
        with torch.no_grad():
            for i, (batch_x, batch_x_mark) in enumerate(test_loader):
                print("Batch number: " + str(i) + " of " + str(len(test_loader)))

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                
                ### reconstruction
                outputs = self.model(batch_x, batch_x_mark, None, None)

                ### note that since we do not use masking but end padding without a mask in training, 
                ### the model will predict the entire sequence, even the padded part
                ### this is not a problem, as we can just ignore the padded part
                ### but we have to exclude the padded part from loss calculation
                ### this is done by multiplying the loss with a mask that is determined by 0 values in batch_x
                mask = (batch_x != 0).float()  # assuming 0 is the padding value
                outputs = outputs * mask
                batch_x = batch_x * mask
                ### the masking step is not really relevant
                ### because masking is done in the model itself as well
                ### we leave it for completeness
                
                ### criterion
                ### acquiring an anomaly score for each band helps to identify the most important bands
                ### (not relevant in the current univariate implementation, but could be useful in the future)
                score_each_band = self.anomaly_criterion(batch_x, outputs)
                score_each_band = score_each_band.cpu().numpy()
                ### [batch_size, seq_len, bands], bands: 1

                ### this is the place where we can restore the normal sequence including NA values
                ### here are the steps to do: 
                ### 1. loop through all the samples in the batch
                ### 2. get the positions of the NA values in the original raster stack/time series
                ###    (we can use all_boas in conjunction with i from the for loop, and iter_locations for this)
                ### 3. remove the padding values (zeros) from the anomaly score time series
                ### 4. add the NA values to the anomaly score time series at the correct location
                ###    (we should end up with a time series of the same length as the original raster stack and the dates list)
                ### 5. write the anomaly score time series to the raster stack
                ### 6. write the anomaly score raster stack to disk
                ###    (each time step/raster separately)
                            
                ### write the restored anomaly scores to the raster stack all_boas
                ### at the correct locations given by i and iter_locations
                for iter in range(batch_x.shape[0]):
                    # print("Sample number: " + str(iter) + " of " + str(batch_x.shape[0]))
                    original_idx = iter_locations[i * self.args.batch_size + iter]
                    ts = score_each_band[iter]
                    ### remove the 0.0 values from ts
                    ### in case the corresponding observation from batch_x is also 0.0
                    ### the observation is given as batch_x[iter, :, :]
                    batch_x_ts = batch_x[iter, :, :].cpu().numpy()
                    valid_mask = ~(np.all(batch_x_ts == 0, axis=1) & np.all(ts == 0, axis=1))
                    ts = ts[valid_mask]
                    ### write the time series ts on the correct location in all_boas
                    ### which is given by all_boas[original_idx, :, :] 
                    ### however, only overwrite the non-nan values
                    ### the nan values should stay at exactly the same location in the sequence
                    ### as they were before
                    ### acquire original_ts (observations) for the current pixel
                    original_ts = all_boas[original_idx, :, :]
                    ### get the positions of the NA values in the original raster stack/time series
                    original_na_positions = np.isnan(original_ts[:, 0])
                    ### initialize an array to store the restored anomaly scores
                    restored_score_ts = np.full(original_ts.shape, np.nan)
                    ### if there is a mismatch in the first dimension of ts and restored_score_ts[~original_na_positions, :], 
                    ### skip the pixel (just a precaution)
                    if ts.shape[0] == restored_score_ts[~original_na_positions, :].shape[0]:
                        ### fill in the anomaly scores at the correct locations
                        restored_score_ts[~original_na_positions, :] = ts
                    ### write the restored anomaly scores to the raster stack
                    all_boas[original_idx, :, 0] = np.squeeze(restored_score_ts)
                    
        print('Predicting finished!')
        print("Current time:")
        print(datetime.now())

        ### all_boas now contains the final anomaly scores
        ### as a raster stack (time series of rasters)
        ### write the anomaly scores to disk
        ### but each time step separately
        ### writing one geotiff file for each time step
        ### naming of the geotiff files: anomaly_scores_YYYYMMDD.tif
        ### the dates list contains the dates in the correct order
        ### remember that all_boas is a numpy array and does not contain geospatial information
        ### the geospatial information can be taken from forest_mask, which is a raster file
        ### and has the same shape as all_boas (except the time steps)
        ### how about using forest_mask as a template for each time step?
        ### also consider that we need to reshape all_boas to the original width and height, 
        ### because it was flattened before
        all_boas = all_boas.reshape((h, w, len(dates), len(self.args.indices_bands)))
        ### save to disk
        for t in range(len(dates)):
            output_path = os.path.join(self.args.result_tile_base_path, self.args.tile_name, f'anomaly_scores_{dates[t].strftime("%Y%m%d")}.tif')
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(all_boas[:, :, t, 0], 1)
        
        print('FORCE tile ' + self.args.tile_name + ' finished and written to disk! Exiting...')
        print("Current time:")
        print(datetime.now())


