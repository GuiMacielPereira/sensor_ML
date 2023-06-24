### Data Folder

The raw data is stored in csv files in the folder raw_csv. Each csv file in this folder corresponds to a song played on GuitarHero.

The script scripts/five_users_to_npz.py takes the data from raw_csv and merges the data from these runs into a single npz file, stored under raw_npz.

The notebook notebooks/01_data_processing.ipynb takes in the data from the npz file and extracts the triggers from the raw signals. The triggers are stored under the folder processed/

The remaining notebooks use the processed data.
