##### this script creates a suitable forest mask for Luxembourg for the model inference
##### we make predictions on all forest pixels of Sentinel-2 in the LUX datacube
##### forest pixels are determined by the Land Use map of Luxembourg of the year 2018, 
##### which is most suitable for the study period of 2017 - 2020

### specify your packages
my_packages <- c('terra', 'foreach', 'sf', 'furrr')
### extract not installed packages
not_installed <- my_packages[!(my_packages %in% installed.packages()[ , 'Package'])]
### install not installed packages
if(length(not_installed)) install.packages(not_installed)

### imports
require(terra)
require(foreach)
require(sf)
require(furrr)

### define the function to process each tile
process_tile <- function(tile, force_tiles_base_path, forest_vect, forest_mask_save_path) {
  print(tile)
  
  ### load the first tif file in the tile folder
  force_tile_list <- list.files(paste0(force_tiles_base_path, '/', tile), pattern = "SEN2A", full.names = TRUE)
  
  ### check if any Sentinel-2 scenes are available in this tile
  if (length(force_tile_list) > 0) {
    print("length of tile list > 0")
    
    force_tile <- rast(force_tile_list[1])$BLUE
    
    ### set all values to 1 to make sure all pixels are considered
    values(force_tile) <- 1
    
    ### reproject the force tile raster to the crs of the forest vector
    force_tile <- terra::project(force_tile, crs(forest_vect))
    
    ### only continue if the force_tile and forest_vect overlap (using terra::relate)
    if (sum(terra::is.related(forest_vect, vect(ext(force_tile)), "intersects")) > 0) {
      print("force tile and forest vector intersect (at least one point in common)")
      
      forest_vect_agg <- aggregate(forest_vect)
      
      ### crop the forest vector to the extent of the force tile
      forest_vect_tile <- crop(forest_vect_agg, vect(ext(force_tile)))
      
      ### mask the force_tile raster with the forest_vect_agg vector
      force_tile <- terra::mask(force_tile, forest_vect_tile, inverse = FALSE)
      
      ### count forest pixels
      count <- global(force_tile, fun = "sum", na.rm = TRUE)
      
      ### save to disk if at least one forest pixel present
      if ((count$sum > 0) && !(is.na(count$sum))) {
        ### create folder structure
        dir.create(paste0(forest_mask_save_path, "/", tile), showWarnings = FALSE, recursive = TRUE)
        
        ### save the forest mask
        writeRaster(force_tile, paste0(forest_mask_save_path, "/", tile, '/forest_mask.tif'), overwrite = TRUE)
      }
    }
  }
}


### define (absolute) paths and other parameters
### base path to the FORCE tiles 
### (i.e. containing the tile folders such as 'X0058_Y0047', which contain the Sentinel-2 data)
force_tiles_base_path <- ''
### base path to the forest mask, also containing the tile folders such as 'X0058_Y0047'
forest_mask_save_path <- ''
no_cpus <- 40

### prepare the land use file (if not done yet)
# LAND_USE_DIR
LAND_USE_VECT <- '/path/to/lu-2007-2015-2018-shp/LU_2007_2015_2018_shp/LU_2007_2015_2018.shp'
# path to prepared land use vector file (if done already)
LAND_USE_VECT_PREPARED <- '/path/to/lu-2007-2015-2018-shp/LU_2007_2015_2018_shp/LU_2007_2015_2018_prepared.shp'

### if the file LAND_USE_VECT_PREPARED does not exist, we execute the following code
if (!file.exists(LAND_USE_VECT_PREPARED)) {
    ### read the land cover file (raster)
    lu18 <- vect(LAND_USE_VECT)
    ### filter for relevant land use class:
    ### LU_2018 == 311, L1_code == 3, L2_code == 31, L3_code == 311 -> forest - forest block - coniferous
    ### do we also accept mixed forest? that would be 312
    ### deciduous forest == 313
    ### young forest == 314
    ### we should add them and perhaps remove later if needed
    ### data also contains forest disturbances: 
    ### 3, 32, 321 -> forest - clearing - burnt area
    ### 3, 32, 323 -> forest - clearing - clear cuts
    selected_values <- c(311, 312, 313, 314, 321, 323)
    lu18 <- lu18[lu18$LU_2018 %in% selected_values, ]

    ### aggregate by field
    lu18 <- aggregate(lu18, by = "LU_2018")

    ### reproject
    lu18 <- terra::project(lu18, "EPSG:3035")

    ### remove some fields
    lu18 <- lu18[, c("LU_2018")]

    ### save the prepared land cover raster
    writeVector(lu18, LAND_USE_VECT_PREPARED, overwrite = TRUE)
}


### we do the following: 
### 1) loop through the grid folders (FORCE tiles) of grid_path
### 2) check if there is an overlap of the FORCE tile with the land use file (check crs!)
### 3) if no, dismiss the FORCE tile
### 4) if yes, create a forest mask for the FORCE tile
### 5) save the forest mask to disk in a specified folder with the same tile (folder) structure as the FORCE tiles
###    the forest mask should be a raster with the same extent and resolution as the FORCE tile

### get tile grid
tiles <- list.dirs(force_tiles_base_path, full.names = FALSE)
### remove the first two entries (the parent folder and the mosaic folder)
tiles <- tiles[-c(1, 2)]

### load forest raster (created a vrt from it beforehand using terra package)
forest_vect <- vect(LAND_USE_VECT_PREPARED)

### set up parallel processing
plan(multicore, workers = no_cpus)

### use future_map to parallelize the processing of tiles
future_map(tiles, process_tile, force_tiles_base_path, forest_vect, forest_mask_save_path)
