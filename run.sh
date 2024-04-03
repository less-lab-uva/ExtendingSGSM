#!/bin/bash

conda activate sg_monitor

### Pull the data
source download_data.sh

### Check the properties
echo Checking InterFuser...
python ./check_properties.py --folder_to_check ./Interfuser_data/ --save_folder ./Interfuser_results/
echo Checking TCP...
python ./check_properties.py --folder_to_check ./TCP_data/ --save_folder ./TCP_results/
echo Checking LAV...
python ./check_properties.py --folder_to_check ./LAV_data/ --save_folder ./LAV_results/

### Generate the tables
echo Generating Tables...
python generate_table.py --base_folder ./ --output_folder ./tables
python generate_duration_table.py --base_folder ./ --output_folder ./tables