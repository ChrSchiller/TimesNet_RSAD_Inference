##### this script reads the list of FORCE tiles containing forest in Germany
##### and loops over them to invoke the Python script 08_sits_bert_prediction_on_force_tiles.py
##### to predict the disturbance estimates for each tile

import os
import subprocess
import argparse

if __name__ == "__main__":

    ### create the parser
    parser = argparse.ArgumentParser()

    ### path to inference script for single tile
    parser.add_argument('--inference_script_path', type=str, required=False, default="inference_on_single_force_tile.py")

    ### path to the model (trained on LUX study site as spatial hold-out, meaning that it is most representative for Germany)
    parser.add_argument('--model_path', type=str, required=False, default="/mnt/storage/forest_decline/paper_anomaly_detection/4_dl_europe_iter7_timesnet_qai_20250127_fullyears/models/anomaly_detection_RsTs_TimesNet_RsTs_RsTs_ftM_sl200_ll48_pl0_dm16_nh8_el2_dl1_df8_fc1_eblearnedsincos_dtTrue_test_con_CRSWIR_final_seed789_0/checkpoint.pth")

    ### force tile base path
    ### default directory is the default directory the FORCE datacube would have on eo-lab.org
    parser.add_argument('--force_tile_base_path', type=str, required=False, default="/media/cangaroo/Elements/christopher/future_forest/forest_decline/force/level2/germany")

    ### result tile base path to store the tile directories and the results
    parser.add_argument('--result_tile_base_path', type=str, required=False, default="/mnt/storage/forest_decline/paper_anomaly_detection/4_dl_europe_iter7_timesnet_qai_20250127_fullyears/models/anomaly_detection_RsTs_TimesNet_RsTs_RsTs_ftM_sl200_ll48_pl0_dm16_nh8_el2_dl1_df8_fc1_eblearnedsincos_dtTrue_test_con_CRSWIR_final_seed789_0/inference")

    ### forest mask base path containing the tile folders and the forest mask.tif
    ### forest mask was created by R script create_forest_mask_by_force_tiles.R
    parser.add_argument('--forest_mask_base_path', type=str, required=False, default="/mnt/storage/forest_decline/inference/forest_mask")

    ### end date of time series
    ### as a string in the format YYYYMMDD
    parser.add_argument('--end_date', type=str, required=False, default="20221231")

    ### parse the arguments
    args = parser.parse_args()

    ### get the tiles
    tiles = os.listdir(args.forest_mask_base_path)
    tiles = [tiles[tile] for tile in range(0, len(tiles))]
    # ### if two VM's are used, just split into two parts: 
    # tiles = tiles[238:len(tiles)] # second half of FORCE tiles
    ### we define the tiles for our study area here
    tiles = ["X0072_Y0049", "X0071_Y0049", "X0071_Y0048"]

    ### loop over tiles
    for tile in tiles:
        print('Next tile to process: ' + tile)

        ### invoke the Python script using the defined arguments
        command = ['python3', args.inference_script_path, 
                '--tile_name', tile, 
                '--model_path', args.model_path, 
                '--force_tile_base_path', args.force_tile_base_path, 
                '--result_tile_base_path', args.result_tile_base_path, 
                '--forest_mask_base_path', args.forest_mask_base_path, 
                '--end_date', args.end_date]

        ### execute the subprocess
        subprocess.call(command)

        print('Finished processing tile ' + tile)
