import numpy as np
import rasterio

def read_mask_force_rasters(file_path, mask, band_list, mean_list, std_list):
    with rasterio.open(file_path) as src:
        ### access metadata to get band names if available
        band_names = src.descriptions  # This will give a tuple of band descriptions (names)

        ### create a dictionary to map band names to their indices
        band_name_to_index = {name: i + 1 for i, name in enumerate(band_names)}

        ### create a list of band indices to keep based on the provided band names
        band_indices_to_keep = [band_name_to_index[name] for name in band_list]

        ### read the specified bands
        rast = src.read(band_indices_to_keep).astype(float)

    ### also read and use the QAI layer of each file
    ### here is the best option because we can save on memory
    qai_path = file_path.replace('BOA', 'QAI')
    with rasterio.open(qai_path) as qai_src:
        qai = qai_src.read().astype(np.uint16)

    ### currently, qai is of shape (1, width, height)
    ### the first dimension is the qai band
    ### it contains integers that can be translated to binary
    ### and then subdivided into multiple categories indicating the quality of the observation
    ### this is what we need to do here
    ### first, convert to 15-bit binary
    ### then, extract individual bits and bit groups using bitwise operations
    ### bit 0 is the right(!!)-most bit in the 16-bit integer (after converting to binary)
    ### (1 << 0) creates a 16-bit integer with all bits set to 0 except the right-most bit
    ### the bits are then shifted to the right to get the value of the bit (e.g. ">> 1" operation)
    valid = (qai & (1 << 0)) >> 0
    cloud = (qai & (3 << 1)) >> 1
    cloud_shadow = (qai & (1 << 3)) >> 3
    snow = (qai & (1 << 4)) >> 4
    water = (qai & (1 << 5)) >> 5
    aerosol = (qai & (3 << 6)) >> 6
    subzero = (qai & (1 << 8)) >> 8
    saturation = (qai & (1 << 9)) >> 9
    high_sun_zenith = (qai & (1 << 10)) >> 10
    illumination = (qai & (3 << 11)) >> 11
    # slope = (qai & (1 << 13)) >> 13
    # water_vapor = (qai & (1 << 14)) >> 14

    ### convert all -9999 values to NA in rast
    rast[rast == -9999] = np.nan

    ### create the filter based on the extracted quality information
    mask_filter = (
        (valid == 1) |
        (cloud == 2) | (cloud == 3) |
        (cloud_shadow == 1) |
        (snow == 1) |
        (water == 1) |
        (aerosol == 2) | (aerosol == 3) |
        (subzero == 1) |
        (saturation == 1) |
        (high_sun_zenith == 1) |
        (illumination == 2) | (illumination == 3)
    )

    ### broadcast the mask_filter to match the shape of rast (more bands)
    mask_filter = np.broadcast_to(mask_filter, rast.shape)

    ### apply the filter
    rast[mask_filter] = np.nan

    ### compute the CRSWIR index to further reduce the size of the raster 
    ### it uses the three bands from band_list
    ### the resulting raster should only have one band: CRSWIR
    ### the equation is: CRSWIR = SW1 / NIR + ((SW2 - NIR) / (2185.7 - 864)) * (1610.4 - 864))
    ### where NIR is band 0, SW1 is band 1, and SW2 is band 2 in the raster file
    # Compute the CRSWIR index
    # NIR = rast[0]
    # SW1 = rast[1]
    # SW2 = rast[2]
    NIR = rast[band_list.index('NIR')]
    SW1 = rast[band_list.index('SWIR1')]
    SW2 = rast[band_list.index('SWIR2')]
    CRSWIR = SW1 / (NIR + ((SW2 - NIR) / (2185.7 - 864)) * (1610.4 - 864))

    ### normalize CRSWIR by mean_list and std_list
    ### note that these are lists (in CRSWIR case of length 1) to be consistent with the training script
    ### X[columns_to_normalize] = X[columns_to_normalize].apply(lambda x: (x - self.overall_mean) / self.overall_std, axis=1)
    CRSWIR = (CRSWIR - mean_list[0]) / std_list[0]

    ### convert all inf and -inf values in CRSWIR to NaN
    CRSWIR[CRSWIR == np.inf] = np.nan
    CRSWIR[CRSWIR == -np.inf] = np.nan

    ### convert numpy array to memory-mapped file
    memmap = np.memmap('rast.mmap', CRSWIR.dtype, mode='w+', shape=CRSWIR.shape)
    ### write the numpy array to the memory-mapped file
    memmap[:, :] = CRSWIR

    # ### convert numpy array to memory-mapped file
    # memmap = np.memmap('rast.mmap', rast.dtype, mode='w+', shape = rast.shape)
    # # write the numpy array to the memory-mapped file
    # memmap[:, :, :] = rast

    ### apply forest mask
    memmap = memmap * mask
    ### remove all -9999 values coming from quality screening (cloud mask, etc.)
    memmap[memmap == -9999] = np.nan
    ### remove rast from memory
    rast = None

    ### add "band" dimension (1) to the memmap array (something like unsqueeze)
    memmap = np.expand_dims(memmap, axis=0)

    # ### write an example raster to disk to check if the CRSWIR index is correct
    # ###       and also to check if the forest mask is being applied correctly
    # with rasterio.open('/mnt/storage/forest_decline/paper_anomaly_detection/4_dl_europe_iter7_timesnet_qai_20250127_fullyears/models/anomaly_detection_RsTs_TimesNet_RsTs_RsTs_ftM_sl200_ll48_pl0_dm16_nh8_el2_dl1_df8_fc1_eblearnedsincos_dtTrue_test_con_CRSWIR_final_seed789_0/inference/X0058_Y0047/crswir.tif', 'w', driver='GTiff', width=memmap.shape[2], height=memmap.shape[1], count=1, dtype=memmap.dtype, crs=src.crs, transform=src.transform) as dst:
    #     dst.write(memmap[0, :, :], 1)

    # ### also write the filter mask to disk for checking
    # ### convert True and False from mask_filter to 1 and 0
    # mask_filter = mask_filter.astype(np.uint8)
    # with rasterio.open('/mnt/storage/forest_decline/paper_anomaly_detection/4_dl_europe_iter7_timesnet_qai_20250127_fullyears/models/anomaly_detection_RsTs_TimesNet_RsTs_RsTs_ftM_sl200_ll48_pl0_dm16_nh8_el2_dl1_df8_fc1_eblearnedsincos_dtTrue_test_con_CRSWIR_final_seed789_0/inference/X0058_Y0047/mask.tif', 'w', driver='GTiff', width=mask_filter.shape[2], height=mask_filter.shape[1], count=1, dtype=mask_filter.dtype, crs=src.crs, transform=src.transform) as dst:
    #     dst.write(mask_filter[0], 1)
    
    # ### also write NIR, SW1 and SW2 to disk for checking
    # with rasterio.open('/mnt/storage/forest_decline/paper_anomaly_detection/4_dl_europe_iter7_timesnet_qai_20250127_fullyears/models/anomaly_detection_RsTs_TimesNet_RsTs_RsTs_ftM_sl200_ll48_pl0_dm16_nh8_el2_dl1_df8_fc1_eblearnedsincos_dtTrue_test_con_CRSWIR_final_seed789_0/inference/X0058_Y0047/NIR.tif', 'w', driver='GTiff', width=NIR.shape[1], height=NIR.shape[0], count=1, dtype=NIR.dtype, crs=src.crs, transform=src.transform) as dst:
    #     dst.write(NIR, 1)
    # with rasterio.open('/mnt/storage/forest_decline/paper_anomaly_detection/4_dl_europe_iter7_timesnet_qai_20250127_fullyears/models/anomaly_detection_RsTs_TimesNet_RsTs_RsTs_ftM_sl200_ll48_pl0_dm16_nh8_el2_dl1_df8_fc1_eblearnedsincos_dtTrue_test_con_CRSWIR_final_seed789_0/inference/X0058_Y0047/SW1.tif', 'w', driver='GTiff', width=SW1.shape[1], height=SW1.shape[0], count=1, dtype=SW1.dtype, crs=src.crs, transform=src.transform) as dst:
    #     dst.write(SW1, 1)
    # with rasterio.open('/mnt/storage/forest_decline/paper_anomaly_detection/4_dl_europe_iter7_timesnet_qai_20250127_fullyears/models/anomaly_detection_RsTs_TimesNet_RsTs_RsTs_ftM_sl200_ll48_pl0_dm16_nh8_el2_dl1_df8_fc1_eblearnedsincos_dtTrue_test_con_CRSWIR_final_seed789_0/inference/X0058_Y0047/SW2.tif', 'w', driver='GTiff', width=SW2.shape[1], height=SW2.shape[0], count=1, dtype=SW2.dtype, crs=src.crs, transform=src.transform) as dst:
    #     dst.write(SW2, 1)

    return memmap.transpose([1, 2, 0]) # put the band dimension in the last position