cd ../src

xvfb-run -a node index.js
python3 link_scraper.py
python3 handle_zips.py
python3 kaggle_rancher.py