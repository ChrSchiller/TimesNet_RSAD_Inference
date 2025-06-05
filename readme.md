# Early Detection of Bark Beetle Infestations in Central Europe using Deep Learning-based Reconstructions of Irregular Sentinel-2 Time Series: Model Inference on FORCE Datacube in Germany

This repository builds upon the publication and corresponding code repository (https://github.com/ChrSchiller/TimesNet_RSAD) of Schiller et al. (2025) in Forestry: An International Journal of Forest Research. The code can be used to make predictions on all forest pixels of the FORCE datacube tiles of Germany, e.g. on www.eolab.org using GPU-powered virtual machines (VM).  
On two Virtual Machines (VMs) equipped with an Nvidia RTX A6000 GPU, 12 CPUs and 112 GB RAM (this is the hardware we can currently use on www.eo-lab.org), predicting each (coniferous) forest pixel in Germany takes about 4 days when splitting the list of FORCE tiles in half and processing each half on one VM. 

## Citation

If you use this repository, please cite the corresponding paper as follows: 

```
@ARTICLE{schiller2025early, 
    title={Early Detection of Bark Beetle Infestations in Central Europe using Deep Learning-based Reconstructions of Irregular Sentinel-2 Time Series}, 
    author={Schiller, Christopher and May, Johannes and Klinke, Randolf and Fassnacht, Fabian Ewald}, 
    journal={Forestry: An International Journal of Forest Research}, 
    volume={TBD},
    pages={TBD}, 
    year={2025}, 
    publisher={Oxford University Press}}
```

Link to the publication: TBD

## Requirements

To use the code, install Python 3.9 and the packages specified in requirements.txt using the following commands: 

```
sudo apt install -y python3.9 python3.9-venv python3.9-dev
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You will also have to install R (v4.3.1 or higher) and the packages terra, sf and foreach to prepare the forest mask. 

## Inference

To run the inference in a (sequential) loop across all FORCE datacube tiles in Germany, execute the following steps.

### Forest Mask Preparation

Download the Copernicus Land Monitoring Service (CLMS) forest mask and the FORCE datacube grid (https://github.com/CODE-DE-EO-Lab/community_FORCE/tree/main/grid).  
Optionally, you may download an outline for the country of interest (e.g. using NUTS, as in this case) and use it in the following R code (or comment out the corresponding code in the R script).  
Next, fill in the correct paths in the following script and execute the R (v4.3.1 or higher) code `create_forest_mask_by_force_tiles.R` with the correct file paths, e.g.

```
Rscript create_forest_mask_by_force_tiles.R 
```

If you're interested in Luxembourg: they have a very good open data policy and provide exceptional datasets for preparing the forest mask. You may use the script `create_lux_forest_mask_by_force_tiles.R` for that by 1) downloading the relevant data (see also Schiller et al. (2025)), 2) filling in the relevant file paths in the mentioned R script and 3) executing the code as follows: 

```
Rscript create_lux_forest_mask_by_force_tiles.R 
```

### Running the inference code

You can either execute only the python script `inference_on_single_force_tile.py` for a specific tile (using the `--tile_name` argument) or the python script `inference_on_all_force_tiles.py` to run the inference on all tiles.  
In each case, you have the option to use the `--end_date` argument or not: if you include this argument, the time series of four years length will be build with the given end date (in YYYYMMDD format). If left blank, the last date of the FORCE tile's Sentinel-2 data will be taken as end date of the input time series. 

The arguments used in the code are:  
`--tile_name` to specify a tile to be processed (only script `inference_on_single_force_tile.py`)  
`--model_path` containing the absolute path to the trained model  
`--force_tile_base_path` containing the base path to the FORCE tiles  
`--result_tile_base_path` containing the base path for the result tile directories  
`--forest_mask_base_path` containing the base path for the forest mask directories  
`--end_date` containing the requested end dates of the input time series (in YYYYMMDD format; optional argument)
  
Here's an example of running the script `inference_on_single_force_tile` (single FORCE tile prediction) with a specific end date of the input time series and a specific tile: 

```
python inference_on_single_force_tile.py --tile_name X0058_Y0058 --end_date 20240930 [add the path arguments here]
```

And here's an example of running the script `inference_on_all_force_tiles.py` (prediction on all available FORCE tiles containing forest) with the latest Sentinel-2 observation as end date: 

```
python inference_on_all_force_tiles.py [add the path arguments here]
```

For an operational forest monitoring system, the script `inference_on_all_force_tiles.py` can be run regularly, e.g. every five days (revisit time of Sentinel-2) using a cronjob or systemd timer. An example on how to do this (naively, with a specific sleep time assuming a FORCE datacube update time of five days) is given in the next chapter.  

Note that currently the only model that is implemented and applicable here is the one using CRSWIR as the only channel (univariate time series). To change this, major adjustments need to be made in `read_mask_force_rasters.py` and `dataloader.py`. The community is invited to add more (or arbitrary) band combinations. For the time, an extension to do so myself is not planned - sorry for that!  
That said, any desired band combination can be easily 'hard-coded' when adapting the above-mentioned (and some other) scripts of this repo (some of the code that can do this is still present but commented out at the moment). 

### Regular inference using systemd timer

In the following, I provide Linux command line code (tested on Linux Ubuntu 20.04 LTS) for the Linux terminal to establish a systemd timer to execute the code provided above in order to achieve a regular prediction of forest disturbance all over Germany. This provides the basis for establishing a nationwide forest disturbance monitoring system.  
The idea is the following: the inference on two VM's each working on half of the FORCE tiles containing forest in Germany takes a bit more than 4 days, while the revisit time of Sentinel-2 (and update time of the corresponding datacube) is about five days. Thus, the code needs to sleep for 1 day until invoking a new inference round. This is what the following code does.  
Admittedly, the code would optimally crawl the FORCE tiles directories to search for latest updates (Sentinel-2 scene uploads), prepare a list of these tiles sorted by update time and feed this list to `inferece_on_single_force_tiles.py`. In this case, we would come as close to near real-time monitoring as possible. This, however, was beyond the scope of the project. Readers are invited to contribute it to this repository.  

#### Create the log directory

Open a linux terminal and execute the following code, which creates a log directory and grants the correct user rights: 

```
mkdir -p ~/logs_cron
chmod 755 ~/logs_cron
```

#### Create the systemd service file

Open the systemd service file for editing, e.g. using nano: 

```
sudo nano /etc/systemd/system/myjob.service
```

Paste the following configuration in the editor and save to disk: 

```
[Unit]
Description=Run the cronjob

[Service]
Type=simple
ExecStart=/bin/bash -c 'mkdir -p ~/logs_cron && touch ~/logs_cron/output.log && chmod 755 ~/logs_cron && source <absolute/path/to/venv/bin/activate> && <absolute/path/to/run_inference_cronjob.sh>'

[Install]
WantedBy=default.target
```

Note that you have to replace the placeholder paths in `run_inference_cronjob.sh` by your own absolute paths to the python interpreter and `inference_on_all_force_tiles.py`. 

#### Create the systemd timer file

Open the systemd timer file for editing, e.g. using nano: 

```
sudo nano /etc/systemd/system/myjob.timer
```

Paste the following configuration in the editor and save to disk: 

```
[Unit]
Description=Run the next job 1 day after the last job finishes

[Timer]
OnUnitActiveSec=1d
Persistent=true

[Install]
WantedBy=timers.target
```

#### Get the service running

Reload the systemd configuration to apply the new service and timer files: 

```
sydo systemctl daemon-reload
```

Start the service and enable the timer to start automatically when booting the computer: 

```
sudo systemctl start myjob.service
sudo systemctl enable myjob.timer
```

Optionally, verify that the setup is running and monitor it:

```
sudo systemctl status myjob.service
journalctl -u myjob.service
sudo systemctl status myjob.timer
```

If you want to disable the timer to prevent it from triggering the service after the current job has finished, you can do it with the following command: 

```
sudo systemctl disable myjob.timer 
```

Alternatively, to stop the running service immediately:

```
sudo systemctl stop myjob.service 
```

## Investigating the results

While depending on your use case, you will likely need to acquire the time series of reconstruction errors for your pixels of interest in order to search for anomalies.  
We've prepared some R scripts to support you with that: `./result_analysis/01_extract_first_anomaly.r` and `./result_analysis/02_plot_example_time_series_results.r`. Here are some quick notes on the scripts: the main input you will need is a metafile of your target Sentinel-22 pixels as polygons, or points/centroids in those polygons. There should be a field serving as an id for the pixel (in the scripts, it is called `plotID`), and optimally a field that denotes the year of the first known disturbance/anomaly (called `earliest` in the script).  
If you don't have the latter information, the script `01_extract_first_anomaly.r` also provides a function that works without temporal information about the disturbance (`02_plot_example_time_series_results` does not need that information anyway). Obviously, this function cannot return the detection delay then, but it will still return the first anomaly detected by the model.  
Here we go (fill in the file paths in the scripts before executing the code): 
```
Rscript ./result_analysis/01_extract_first_anomaly.r
Rscript ./result_analysis/02_plot_example_time_series_results.r
```

## Acknowledgments

This code is strongly based on the TimesNet repository (https://github.com/thuml/TimesNet), as we adjusted the TimesNet model for our publication. All credits to the authors of this publication and repository. Thank you very much!

Additionally, I want to thank Max Freudenberg from University of Göttingen for many code snippets used in this repository, e.g. reading and subsetting the list of FORCE tile rasters efficiently as well as stacking the raster. Thank you very much, Max! 