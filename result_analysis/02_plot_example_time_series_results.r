###### this script is supposed to plot example time series of the anomaly scores
###### for some selected pixels (e.g. random sample of each detected year, or even selected by single pixel's plotID)


### load libraries
require(terra)
require(sf)
require(dplyr)
require(ggplot2)
require(future)
require(future.apply)


##### define functions

### write a function that plots the results as a time series
### the csv files for the time series are stored in paste0(force_tiles_base_path, "/results/ts") and named after the plotID
### some of these files might be empty, so the function needs to skip those samples without any observations
plot_time_series <- function(sample, threshold) {
  plotID <- sample$plotID
  csv_file <- paste0(force_tiles_base_path, "/results/ts/", plotID, ".csv")
  
  ### check if the CSV file exists and is not empty
  if (file.exists(csv_file) && file.info(csv_file)$size > 0) {
    data <- read.csv(csv_file)
    
    ### check if the data has any rows
    if (nrow(data) > 0) {
      ### create the plot
      p <- ggplot(data, aes(x = as.POSIXct(date, format = "%Y-%m-%d"))) +
        geom_line(aes(y = value), color = "black", linewidth = 1) +
        geom_ribbon(aes(ymin = 0, ymax = value), fill = "gray80", alpha = 0.5) +
        geom_hline(yintercept = threshold, linetype = "dashed", color = "gray40", linewidth = 0.5) +  # Add horizontal dashed line at threshold
        scale_x_datetime(date_breaks = "2 months", date_labels = "%Y-%m") + 
        scale_y_continuous(
          name = "Anomaly Score (Squared Error)"
        ) +
        labs(x = "Date") +
        theme_bw() +
        theme(
          axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), 
          text = element_text(size = 18),  # Increase font size for all text
          axis.title = element_text(size = 18),  # Increase font size for axis titles
          axis.text = element_text(size = 16),  # Increase font size for axis labels
          legend.text = element_text(size = 18),  # Increase font size for legend text
          legend.title = element_text(size = 20),  # Increase font size for legend title
          plot.margin = unit(c(1, 1, 1, 1), "cm")  # Adjust plot margins
        )
      
      ### save the plot as a PDF
      ggsave(paste0(force_tiles_base_path, "/results/plot_example_ts/", plotID, "_ref_year_", as.character(sample$year), "_detect_", as.character(sample$first_anomaly_qgis), ".pdf"), p, width = 10, height = 7)
    }
  }
}


### define relevant paths and parameters
### base path for the FORCE tiles (same path as in preceding R script)
force_tiles_base_path <- ''
### decision threshold determined by the user (examples are from Schiller et al. (2025): threshold factors 1.5 and 1 from best seed)
threshold <- 1.6659 # 1.138
### number of workers for parallel processing
num_workers <- 1
### path to the metadata file saved in the preceding script
meta_path <- paste0(force_tiles_base_path, "/results/tmp/", "pixelbased_merged_first_anomalies.gpkg")
### assign number of samples per observation year to be plotted
samples_per_year <- 100

### read file
meta <- vect(meta_path)

### create plotting directory if it does not exist
dir.create(paste0(force_tiles_base_path, "/results/plot_example_ts"), showWarnings = FALSE, recursive = TRUE)

### make a selection of pixels by random selection (random sample by year)
### firstly: random selection of samples_per detection year (given as "first_detection_year" column)
meta_samp <- as.data.frame(meta) %>%
  group_by(first_detection_year) %>%
  sample_n(samples_per_year, replace = FALSE) %>%
  ungroup()
meta_samp <- as.data.frame(meta_samp)

### parallelize the plot_time_series function
plan(multicore, workers = num_workers)
### execute the plot_time_series function for each sample in the meta_samp data frame in parallel
future_lapply(1:nrow(meta_samp), function(i) {
  plot_time_series(meta_samp[i, ], threshold)
})

print("Plots for example time series have been created and saved in the results/plot_example_ts directory. Exiting...")