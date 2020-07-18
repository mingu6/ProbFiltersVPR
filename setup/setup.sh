#!/bin/bash/

read -p "Path of the directory where datasets are stored and read: " dir
echo "RAW_PATH = '$dir/raw/'" >> ./src/settings.py
echo "READY_PATH = '$dir/ready/'" >> ./src/settings.py
echo "PROCESSED_PATH = '$dir/processed/'" >> ./src/settings.py

read -p "Insert RobotCar dataset website username: " name
read -s -p "Insert website password: " pword

python3 ./src/thirdparty/RobotCarDataset-Scraper/scrape_mrgdatashare.py --downloads_dir $dir/raw/ --datasets_file ./src/thirdparty/RobotCarDataset-Scraper/datasets_list.csv --username $name --password $pword
