from pathlib import Path

# Build path by receding from current file
data_dir = Path(__file__).parent.parent.parent / "data"

# Set data path for processed data for 5 users
datapath_five_Path = data_dir / "processed" / "five_users_main_collection_window_32.npz"
datapath_five_users = str(datapath_five_Path)

datapath_three_Path = data_dir / "processed" / "three_users_data_window_32.npz"
datapath_three_users = str(datapath_three_Path)
