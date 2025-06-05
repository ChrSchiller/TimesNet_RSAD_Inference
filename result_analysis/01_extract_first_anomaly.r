##### this script takes a metafile of S2 pixels (polygons)
##### converts them to centroids and then uses the metafile
##### to extract information from a set of rasters
##### the script works and is tested on tiles of a datacube of Sentinel-2 data preprocessed with FORCE (Frantz 2019)
##### in detail: 
##### 1. read the metafile and convert polygons to centroids
##### 2. loop through the three FORCE tiles of the study area in a sequential loop
##### 3. for each FORCE tile, stack the rasters in the tile and get a vector of dates from the file names
##### 4. write a function that extracts the date of the first time two consecutive observations from the raster stack of the same year exceed a certain threshold
##### 5. parallelize this function and apply it to the centroids of the FORCE TILE
##### 6. add the dates to the metafile

### specify your packages
my_packages <- c('ggplot2', 'tidyr', 'terra', 'sf', 'future', 'stringr',
                 'future.apply', 'purrr', 'dplyr', 'lubridate')
### extract not installed packages
not_installed <- my_packages[!(my_packages %in% installed.packages()[ , 'Package'])]
### install not installed packages
if(length(not_installed)) install.packages(not_installed)


### load libraries
library(terra)
library(sf)
library(future)
library(future.apply)
library(dplyr)
library(tidyr)
library(purrr)
library(stringr)
library(lubridate)
library(ggplot2)


##### define functions

### write function that reads through the "meta" file (points) and extracts all values from the raster stack as a time series
### result will have two columns: time series values (anomalies) and corresponding dates
### store it as csv file in "ts" directory using the plotID as file name
### the function can and should be parallelized
extract_write_ts <- function(x, rasters, dates) {
    ### input to the function is a single point of a spatvector (terra package),
    ### the raster stack containing the values to be analysed
    ### and the dates of the rasters
    ### the function returns a data frame with two columns: time series values (anomalies) and corresponding dates
    ### stores it as csv file in "ts" directory using the plotID as file name
    ### so we need to extract the values for this point from the raster stack
    ### and then write it to disk
    values <- terra::extract(rasters, x)
    ### remove ID column
    values <- values[-1]
    ### remove all NA columns but keep column names to identify the dates
    dates <- dates[!is.na(values)]
    values <- values[!is.na(values)]
    ### create data frame
    df <- data.frame(date = dates, value = values)
    ### write to disk
    write.csv(df, paste0(force_tiles_base_path, "/results/ts/", x$plotID, ".csv"), row.names = FALSE)

### end of "extract_write_ts" function
}


### write function to extract first anomaly
### meaning the first time two consecutive observations from the raster stack of the same year exceed a certain threshold
extract_first_anomaly <- function(x, threshold, rasters, dates, years) {
    ### input to the function is a single point of a spatvector (terra package),
    ### a threshold, the raster stack containing the values to be analysed
    ### and the dates of the rasters
    ### the function returns the date of the first time two consecutive observations
    ### from the raster stack of the same year exceed the threshold
    ### so we need to extract the year from the dates, 
    ### then loop through the years, extract the relevant values (time series) for this point
    ### and check the condition
    first_detection <- NA
    for (year in years) {
        ### subset the raster stack to the layers of the year
        ### we can do that by identifying the indices from the year in the dates vector
        ### and then subset the raster stack with these indices
        year_indices <- which(substr(dates, 1, 4) == year)
        rasters_year <- rasters[[year_indices]]
        ### extract the values for the point
        values <- terra::extract(rasters_year, x)
        ### remove ID column
        values <- values[-1]
        ### remove all NA columns but keep column names to identify the dates
        dates_year <- dates[year_indices]
        dates_year <- dates_year[!is.na(values)]
        values <- values[, !is.na(values)]
        ### if there are less than two values, we cannot have two consecutive observations
        ### so we skip this year
        if (length(values) < 2) {
            next
        }
        ### loop through the values and check the condition
        ### if a suitable date is found, do not loop further through the years
        ### (our goal is not the first anomaly per year, but in total/over all years)
        ### note that if you need the first anomaly per year, you can easily change the following code snippet
        ### to return the first detection per year
        for (i in 1:(length(values)-1)) {
            if (values[i] > threshold && values[i+1] > threshold) {
                ### the return function makes sure that the first detection across all years 
                ### is returned (return always breaks the loop)
                first_detection <- as.character(dates_year[i])
                return(first_detection)
            }
        }
    ### end of for loop through years
    }

    ### if first_detection is still NA, return it as NA
    return(first_detection)

### end of function
}

### write function to extract the first detection after an anomaly
### meaning the first time two consecutive observations from the raster stack of the same year exceed a certain threshold
### note that this function is similar to the previous one, but it takes an additional parameter "earliest"
### it is specifically designed to find the first detection after a known anomaly
### instead of the first detection in general
extract_first_detection_after_anomaly <- function(x, threshold, rasters, dates, earliest) {
    ### input to the function is a single point of a spatvector (terra package),
    ### a threshold, the raster stack containing the values to be analysed,
    ### the dates of the rasters, and the year of the known anomaly (earliest)
    ### the function returns the date of the first time two consecutive observations
    ### from the raster stack of the same year exceed the threshold
    ### so we need to extract the year from the dates,
    ### then loop through the years, extract the relevant values (time series) for this point
    ### and check the condition
    years <- as.character(seq(as.numeric(earliest), 2022))
    first_detection <- NA
    for (year in years) {
        ### subset the raster stack to the layers of the year
        ### we can do that by identifying the indices from the year in the dates vector
        ### and then subset the raster stack with these indices
        year_indices <- which(substr(dates, 1, 4) == year)
        rasters_year <- rasters[[year_indices]]
        ### extract the values for the point
        values <- terra::extract(rasters_year, x)
        ### remove ID column
        values <- values[-1]
        ### remove all NA columns but keep column names to identify the dates
        dates_year <- dates[year_indices]
        dates_year <- dates_year[!is.na(values)]
        values <- values[, !is.na(values)]
        ### if there are less than two values, we cannot have two consecutive observations
        ### so we skip this year
        if (length(values) < 2) {
            next
        }
        ### loop through the values and check the condition
        ### if a suitable date is found, do not loop further through the years
        ### (we don't need the first anomaly per year, but in total)
        for (i in 1:(length(values)-1)) {
            if (values[i] > threshold && values[i+1] > threshold) {
                ### the return function makes sure that the first detection across all years 
                ### is returned (return always breaks the loop)
                first_detection <- as.character(dates_year[i+1])
                return(first_detection)
            }
        }
    ### end of for loop through years
    }

    ### if first_detection is still NA, return it as NA
    return(first_detection)

### end of function
}

##### define paths and parameters
### decision threshold determined by the user (examples are from Schiller et al. (2025): threshold factors 1.5 and 1 from best seed)
threshold <- 1.6659 # 1.138
### base path for the FORCE tiles
force_tiles_base_path <- ''
### names of the tiles to be processed
tile_names <- c('X0071_Y0048', 'X0071_Y0049', 'X0072_Y0049')
### number of workers for parallel processing
num_workers <- 1
### path to the original metafile containing the target Sentinel-2 pixels as polygons
### (centroids are also okay, but then centroids(meta) might fail (see below))
### while centroids are good practice, the location of the points only needs to be clearly within the pixel
original_meta_path <- ""
### the meta file needs to contain the columns "plotID", ("mort_2") and "earliest", 
### the latter being the year of the earliest disturbance
### if this time is not available, use extract_first_anomaly function instead of extract_first_detection_after_anomaly
### it should work without the earliest column, but then you obviously cannot get information about the detection delay
### the plotID is used to identify the points in the rasters and to write the time series to disk
### the metafile may contain pixels from different FORCE tiles - the script will handle this
### the code assumes that the metafile is in the same coordinate reference system as the rasters
### if the metafile is not in the same CRS as the rasters, you need to reproject it first
### the mort_2 column is optional, but if it is present, it will be used to filter the points later on
### it represents the fraction to which the pixel is disturbed by the bark beetle infestation
### if the mort_2 column is not present, the script will work, but skip this analysis step
### if the mort_2 column is present, the script will filter the points to only those with mort_2 == 1 (fully disturbed) 
### for some refined analyses



### read metafile
meta <- vect(original_meta_path)

### get centroids (if necessary)
meta <- centroids(meta)

### create directory for output in force_tiles_base_path
dir.create(paste0(force_tiles_base_path, "/results/tmp"), showWarnings = FALSE, recursive = TRUE)

### for loop through tile names
for (tile_name in tile_names) {
    print(tile_name)
  
    ### read rasters
    rasters <- list.files(file.path(force_tiles_base_path, tile_name), full.names = TRUE, pattern = 'tif')
    rasters <- rast(rasters)

    ### get dates from the file names
    file_names <- list.files(file.path(force_tiles_base_path, tile_name), full.names = FALSE, pattern = 'tif')
    dates <- substr(file_names, nchar(file_names)-11, nchar(file_names)-4)
    dates <- as.Date(dates, format = '%Y%m%d')

    ### remove all winter observations (between 1 January and 15 March, and between 15 November and 31 December)
    ### from raster stack and from dates
    winter_dates <- dates[(month(dates) == 1) | (month(dates) == 2) | (month(dates) == 3 & day(dates) < 15) | (month(dates) == 11 & day(dates) > 15) | (month(dates) == 12)]
    winter_indices <- which(dates %in% winter_dates)
    rasters <- rasters[[-winter_indices]]
    dates <- dates[-winter_indices]

    ### create directory for time series output
    dir.create(paste0(force_tiles_base_path, "/results/ts"), showWarnings = FALSE, recursive = TRUE)

    ### crop meta file by extent of one of the raster layers
    meta_tile <- terra::crop(meta, ext(rasters[[1]]))

    # ### write to disk for checking
    # writeVector(meta_tile, paste0(force_tiles_base_path, "/results/tmp/", tile_name, "_meta_tile_after_cropping.gpkg"), overwrite = TRUE)

    ### parallelize extract_write_ts function
    plan(multicore, workers = num_workers)
    future_lapply(1:nrow(meta_tile), function(i) {
        extract_write_ts(meta_tile[i, ], rasters, dates)
    })
    
    # ### following function is currently commented out, 
    # ### because we are not searching for the first detection (which would be model evaluation), 
    # ### but merely the first detection after the anomaly (refining the temporal resolution of anomaly detection)
    # ### parallelize extract_first_anomaly function
    # plan(multicore, workers = num_workers)
    # years <- c("2017", "2018", "2019", "2020", "2021", "2022")
    # first_anomalies <- future_lapply(1:nrow(meta_tile), function(i) {
    #     extract_first_anomaly(meta_tile[i, ], threshold, rasters, dates, years)
    # })

    ### parallelize extract_first_detection_after_anomaly function
    plan(multicore, workers = num_workers)
    first_anomalies <- future_lapply(1:nrow(meta_tile), function(i) {
        extract_first_detection_after_anomaly(meta_tile[i, ], threshold, rasters, dates, meta_tile[i, ]$earliest)
    })
    
    ### add dates to metafile
    meta_tile$first_anomaly <- first_anomalies
    table(meta_tile$first_anomaly, exclude = FALSE)
    
    ### percentage of detected anomalies (not NA)
    ### computation works only like that if the dataset contains only disturbances, no undisturbed pixels
    sum(!is.na(meta_tile$first_anomaly)) / nrow(meta_tile)

    ### print that result incl. tile name
    print(paste0('Percentage of detected anomalies in tile ', tile_name, ': ', sum(!is.na(meta_tile$first_anomaly)) / nrow(meta_tile)))

    ### check in how many cases the year of detection (extracted from meta_tile$first_anomaly)
    ### is equal to the year of the date in meta_tile$earliest
    year_comparison <- substr(meta_tile$first_anomaly, 1, 4) == substr(meta_tile$earliest, 1, 4)
    ### if year_comparison is NA, convert to False
    year_comparison[is.na(year_comparison)] <- FALSE
    sum(year_comparison) / sum(!is.na(meta_tile$first_anomaly))

    ### print that result incl. tile name
    print(paste0('Percentage of anomalies detected in the same year in tile ', tile_name, ': ', sum(year_comparison) / sum(!is.na(meta_tile$first_anomaly))))

    ### now check in how many cases the year of detection is max. one year different from the year of the date in meta_tile$earliest
    year_comparison <- abs(as.numeric(substr(meta_tile$first_anomaly, 1, 4)) - as.numeric(substr(meta_tile$earliest, 1, 4))) <= 1
    ### if year_comparison is NA, convert to False
    year_comparison[is.na(year_comparison)] <- FALSE
    sum(year_comparison) / sum(!is.na(meta_tile$first_anomaly))

    ### print that result incl. tile name
    print(paste0('Percentage of anomalies detected within one year in tile ', tile_name, ': ', sum(year_comparison) / sum(!is.na(meta_tile$first_anomaly))))

    ### convert meta_tile$first_anomaly to character
    ### this is necessary to avoid issues with date formats later on
    meta_tile$first_anomaly <- as.character(meta_tile$first_anomaly)

    ### convert meta_tile$first_anomaly to a format that can be better visualized in QGIS
    ### days could even be dropped, months and years are enough
    ### maybe just YYYYMM
    meta_tile$first_anomaly_qgis <- str_replace(substr(meta_tile$first_anomaly, 1, 7), "-", "")

    ### add a column for first_detection_year
    meta_tile$first_detection_year <- substr(meta_tile$first_anomaly, 1, 4)

    ### add tile name
    meta_tile$tile <- tile_name

    writeVector(meta_tile, paste0(force_tiles_base_path, "/results/tmp/", tile_name, "_first_anomalies.gpkg"), overwrite = TRUE)

    ### check if meta contains a mort_2 column and only then continue with the following code
    if ("mort_2" %in% names(meta_tile)) {
        ### use only the mort_2 == 1 points and check and print the same results as above
        ### (percentage of detected anomalies, percentage of anomalies detected in the same year, percentage of anomalies detected within one year)
        meta_tile_mort2 <- meta_tile[meta_tile$mort_2 == 1, ]
        ### check and also print the results (print command)
        table(meta_tile_mort2$first_anomaly, exclude = FALSE)
        print(paste0('Percentage of detected anomalies in tile ', tile_name, ' (mort_2 == 1): ', sum(!is.na(meta_tile_mort2$first_anomaly)) / nrow(meta_tile_mort2)))
        year_comparison <- substr(meta_tile_mort2$first_anomaly, 1, 4) == substr(meta_tile_mort2$earliest, 1, 4)
        year_comparison[is.na(year_comparison)] <- FALSE
        print(paste0('Percentage of anomalies detected in the same year in tile ', tile_name, ' (mort_2 == 1): ', sum(year_comparison) / sum(!is.na(meta_tile_mort2$first_anomaly))))
        year_comparison <- as.numeric(substr(meta_tile_mort2$first_anomaly, 1, 4)) - as.numeric(substr(meta_tile_mort2$earliest, 1, 4)) <= 1
        # year_comparison <- abs(as.numeric(substr(meta_tile_mort2$first_anomaly, 1, 4)) - as.numeric(substr(meta_tile_mort2$earliest, 1, 4))) <= 1
        year_comparison[is.na(year_comparison)] <- FALSE
        print(paste0('Percentage of anomalies detected within one year in tile ', tile_name, ' (mort_2 == 1): ', sum(year_comparison) / sum(!is.na(meta_tile_mort2$first_anomaly))))

        ## write subset vector
        writeVector(meta_tile_mort2, paste0(force_tiles_base_path, "/results/tmp/", tile_name, "_first_anomalies_fully_disturbed_pixels.gpkg"), overwrite = TRUE)

        ### save a nice histogram of meta_tile_mort2$first_anomaly_qgis in correct order (YYYYMM)
        ### to the same directory as the metafile
        plot <- ggplot(as.data.frame(meta_tile_mort2), aes(x = first_anomaly_qgis)) +
            geom_bar(stat = "count", position = "dodge", color = "black", fill = "darkgray") +
            geom_text(stat = "count", aes(label = ..count..), vjust = -0.5, size = 5) +
            labs(
                x = "First Anomaly (YYYYMM)",
                y = "Number of Pixels"
            ) +
            scale_y_continuous(breaks = seq(0, max(table(meta_tile_mort2$first_anomaly_qgis)), by = 200)) +
            theme_bw() +
            theme(
                legend.position = "none",  # Remove the legend
                axis.title = element_text(size = 24),  # Increase font size for axis titles
                axis.text = element_text(size = 22),  # Increase font size for axis labels
                axis.text.x = element_text(angle = 45, hjust = 1)  # Rotate x-axis labels
            )

        ggsave(paste0(force_tiles_base_path, "/results/tmp/", tile_name, "_first_anomalies_fully_disturbed_pixels_histogram.pdf"), plot, width = 20, height = 10)

    }
    
    
### end of for loop through tile names    
}

### combine the output of the for loop to one meta_combined spatvector of centroids
meta_combined <- vect(paste0(force_tiles_base_path, "/results/tmp/", tile_names[1], "_first_anomalies_fully_disturbed_pixels.gpkg"))
for (tile_name in tile_names[-1]) {
    meta_combined <- rbind(meta_combined, vect(paste0(force_tiles_base_path, "/results/tmp/", tile_name, "_first_anomalies_fully_disturbed_pixels.gpkg")))
}

### provide the performance metrics as in the loop with this file as well
### and print it (percentage of detected anomalies, percentage of anomalies detected in the same year, percentage of anomalies detected within one year)
table(meta_combined$first_anomaly, exclude = FALSE)
print(paste0('Percentage of detected anomalies rel. to all anomalies in all tiles: ', sum(!is.na(meta_combined$first_anomaly)) / nrow(meta_combined)))
year_comparison <- substr(meta_combined$first_anomaly, 1, 4) == substr(meta_combined$earliest, 1, 4)
year_comparison[is.na(year_comparison)] <- FALSE
print(paste0('Percentage of anomalies detected in the same year rel. to all detected anomalies in all tiles: ', sum(year_comparison) / sum(!is.na(meta_combined$first_anomaly))))
year_comparison <- (as.numeric(substr(meta_combined$first_anomaly, 1, 4)) - as.numeric(substr(meta_combined$earliest, 1, 4))) <= 1
# year_comparison <- abs(as.numeric(substr(meta_combined$first_anomaly, 1, 4)) - as.numeric(substr(meta_combined$earliest, 1, 4))) <= 1
year_comparison[is.na(year_comparison)] <- FALSE
print(paste0('Percentage of anomalies detected within one year in all tiles (rel. to all detections, not all pixels): ', sum(year_comparison) / sum(!is.na(meta_combined$first_anomaly))))

### get the percentage of true positives over true positives and false negatives (recall)
### by disturbance year (given in earliest column)
# Group by each year in earliest and calculate the percentage of anomalies detected within one year
results <- as.data.frame(meta_combined) %>%
  mutate(
    first_anomaly_year = as.numeric(substr(first_anomaly, 1, 4)),
    earliest_year = as.numeric(substr(earliest, 1, 4))
  ) %>%
  group_by(earliest_year) %>%
  summarise(
    total = n(),
    detected_within_one_year = sum((first_anomaly_year - earliest_year) <= 1, na.rm = TRUE),
    percentage_detected_within_one_year = detected_within_one_year / total * 100
  )

# Print the results
print(results)

# Calculate the overall percentage of anomalies detected within one year
overall_detected_within_one_year <- sum(results$detected_within_one_year) / sum(results$total) * 100
print(paste0('Overall percentage of anomalies detected within one year (rel. to all disturbances incl. false negatives): ', overall_detected_within_one_year))

### plot the same histogram as above with the current dataset meta_combined
plot <- ggplot(as.data.frame(meta_combined), aes(x = first_anomaly_qgis)) +
    geom_bar(stat = "count", position = "dodge", color = "black", fill = "darkgray") +
    geom_text(stat = "count", aes(label = ..count..), vjust = -0.5, size = 5) +
    labs(
        x = "First Anomaly (YYYYMM)",
        y = "Number of Pixels"
    ) +
    scale_y_continuous(breaks = seq(0, max(table(meta_combined$first_anomaly_qgis)), by = 2000)) +
    theme_bw() +
    theme(
        legend.position = "none",  # Remove the legend
        axis.title = element_text(size = 24),  # Increase font size for axis titles
        axis.text = element_text(size = 22),  # Increase font size for axis labels
        axis.text.x = element_text(angle = 45, hjust = 1)  # Rotate x-axis labels
    )

### save to disk
ggsave(paste0(force_tiles_base_path, "/results/tmp/", "first_anomalies_fully_disturbed_pixels_histogram.pdf"), plot, width = 26, height = 10)



### join the anomaly detection information meta_combined to the original metafile 
### so that we have pixels again (not points)
meta_pix <- vect(original_meta_path)
names(meta_pix)
names(meta_combined)
### remove all but plotID column from meta_pix
meta_pix <- meta_pix[, c('plotID')]

### remove all rows of meta_pix that do not contain a plotID in meta_combined
### this is optional because we can also analyse the situation with all pixels (not only fully disturbed pixels)
meta_pix <- meta_pix[meta_pix$plotID %in% meta_combined$plotID, ]

### join meta_combined information to meta_pix
meta_pix <- terra::merge(meta_pix, meta_combined, by = 'plotID', all.x = TRUE, all.y = TRUE)

### write meta_pix to disk to check it (as geopackage)
writeVector(meta_pix, paste0(force_tiles_base_path, "/results/tmp/", "pixelbased_merged_first_anomalies.gpkg"), overwrite = TRUE)

### aggregate pixels by year of first anomaly
meta_pix_agg <- terra::aggregate(meta_pix, by = 'first_detection_year')

### remove all but the first_detection_year column
meta_pix_agg <- meta_pix_agg[, c('first_detection_year')]

### write to disk
writeVector(meta_pix_agg, paste0(force_tiles_base_path, "/results/tmp/", "pixelbased_merged_first_anomalies_agg.gpkg"), overwrite = TRUE)

print("All done. Exiting...")