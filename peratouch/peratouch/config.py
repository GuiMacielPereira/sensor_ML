from pathlib import Path

# Build path by receding from current file
data_dir = Path(__file__).parent.parent.parent / "data"

# Set data path for processed data for 5 users
path_five_users_processed = str(data_dir / "processed" / "five_users_window_32.npz")

# Path to store analysis results 
path_analysis_results = Path(__file__).parent.parent.parent / "results" / "analysis"

# Path to figures fom analysis
path_analysis_figures = Path(__file__).parent.parent.parent / "figures" / "analysis"

# Path to figures
path_figures = Path(__file__).parent.parent.parent / "figures" 

# REsults
path_results = Path(__file__).parent.parent.parent / "results" 
