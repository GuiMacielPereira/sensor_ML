### How to setup:

`pip install -r requirements.txt`
`pip install -e peratouch`

### Where to start:

Run script `five_users_to_npz.py` to convert raw data into a npz file:

`python3 scripts/five_users_to_npz.py`

Open first notebook `01_data_processing.ipynb` and run all cells to extract trigger sections from data.

### Training the networks 

Run notebook `02_networks.ipynb` to train all networks.

### Results and figures
The results of training the networks are stored under results/
To produce the figures, run the script

`python3 scripts/create_result_plots.py`

and figures will be stored under figures/


