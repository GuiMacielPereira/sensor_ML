## Identifying user presses using Neural Networks built with PyTorch

The raw data for 5 people was recorded using Peratech sensors, and the presses were collected whilst users were playing Guitar Hero 3. The notebooks show the procedure to train three models: CNN, LSTM and CNN-LSTM. The inner workings of these networks and the training procedure can be found inside the folder peratouch.

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


