##### this script creates a forest mask for each FORCE tile in Germany
##### if it contains at least one forest pixel
##### the forest mask is created from the Copernicus Land Monitoring System (CLMS) Forest Layers raster data

### specify your packages
my_packages <- c('terra', 'foreach', 'sf', 'doParallel')
### extract not installed packages
not_installed <- my_packages[!(my_packages %in% installed.packages()[ , 'Package'])]
### install not installed packages
if(length(not_installed)) install.packages(not_installed)

### imports
require(doParallel)
require(terra)
require(foreach)
require(sf)

setwd('working_dir_path')

### get tile grid
grid_path <- '/path/to/datacube-grid_DEU.gpkg'
grid <- vect(grid_path)

### load forest raster (created a vrt from it beforehand using terra package)
forest_mask_vrt_path <- 'path/to/copernicus_forest_layers/germany/DATA/germany.vrt'
forest <- rast(forest_mask_vrt_path)

### need same crs
grid <- terra::project(grid, crs(forest))

### initialize parallelization and foreach loop here
no_cpus <- 40

my.cluster <- parallel::makeCluster(
  no_cpus,
  type = "FORK",
  outfile = ""
)

# register it to be used by %dopar%
doParallel::registerDoParallel(cl = my.cluster)

### foreach loop over tiles
foreach(iter = 1:length(grid)) %dopar% {
### loop through grid (alternatively with a simple for loop)
# for (iter in 1:length(grid)){
  
  ### load forest raster
  forest <- rast(forest_mask_vrt_path)
  
  ### load an reproject grid
  grid <- vect(grid_path)
  grid <- terra::project(grid, crs(forest))
    
  tile <- grid[iter]
  
  ### crop forest cover layer
  forest_tile <- crop(forest, tile)
  
  ### reduce to pure forest mask
  forest_tile[forest_tile == 0, ] <- NA
  forest_tile[forest_tile == 1, ] <- 1 # IMPORTANT: set NA instead of 1 for only coniferous forest
  forest_tile[forest_tile == 2, ] <- 1
  forest_tile[forest_tile > 2, ] <- NA
  
  ### create polygons to be able to buffer the forest with a negative buffer of 15m (forest edges)
  forest_poly <- as.polygons(forest_tile, dissolve = TRUE, na.rm = TRUE)
  forest_poly <- aggregate(forest_poly)
  forest_poly <- vect(st_buffer(st_as_sf(forest_poly), -15))
  
  ### read example raster from FORCE tile
  force_tile_list <- list.files(paste0('/path/to/force/datacube/force/level2/germany/', tile$Tile_ID), 
                                pattern = "SEN2A", full.names = TRUE) 
  # pattern = "SEN2A" so that there cannot be any issue with different origins (as SEN2A and SEN2B can have a tiny mismatch)
  
  ### read example scene
  force_tile <- rast(force_tile_list[1])$BLUE
  
  ### get forest values back to raster format
  ### at the same time, resample to force tile resolution and pixel location
  forest_tile <- rasterize(forest_poly, force_tile, fun = "max")

  ### get outline of Germany for masking
  countries <- vect('/path/to/outline_germany.shp') # e.g. from NUTS
  countries <- countries[countries$NUTS_ID == "DE"]
  countries <- terra::project(countries, crs(forest))

  ### mask by outline of Germany
  forest_tile <- terra::mask(forest_tile, countries, inverse = FALSE)
  
  ### count forest pixels
  count <- global(forest_tile, fun="sum", na.rm=TRUE)
  
  ### save to disk if at least one forest pixel present
  if ((count$sum > 0) & !(is.na(count$sum))){
    
    dir.create(paste0('/path/to/forest_mask/', tile$Tile_ID))
    
    writeRaster(forest_tile, paste0('/output/path/forest_mask/', tile$Tile_ID, '/forest_mask.tif'), overwrite = TRUE)
    
  ### end of if condition regarding valid values
  }
  
### end of for loop over FORCE tiles
}