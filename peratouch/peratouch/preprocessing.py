# Functions used for processing raw data
# i.e. selecting triggers

import numpy as np
from peratouch.plot import plot_grid, plot_X, plot_flatten

class SelectTriggers:

    def __init__(self, data : dict):
        self.data = data
        self.trigger_idxs = {}
        self.clean_triggers = {}
        self.noisy_triggers = {}
        self.short_triggers = {}

    def run(self):
        self.find_idxs()
        self.cut_windows()
        self.print_findings()
        

    def find_idxs(self, zero_threshold=0.05):    # Rather arbitrary definition of the zero line 
        """
        Finds the indexes of the signal for the beggining of trigger
        """
        for key in self.data:
            signal = self.data[key].squeeze()     # Read signal of each user

            zerosIdx = np.argwhere(signal<=zero_threshold)        # Define zero as anything below harcoded value 
            jumpIdx = np.argwhere(np.diff(zerosIdx[:, 0])>1)   # Non-consecutive zeros
            trigIdx =  zerosIdx[jumpIdx]

            self.trigger_idxs[key] = trigIdx.flatten()


    def cut_windows(self, width=32):
        for key in self.data:

            # Initiate empty lists 
            clean_triggers = []
            noisy_triggers = []
            short_triggers = []

            signal = self.data[key].squeeze()

            for i in self.trigger_idxs[key]: 
                trigger = signal[i:i+width]

                if np.mean(trigger<=1)>=0.6:   # If 60% of the signal is below 1 
                    noisy_triggers.append(trigger)
                else:
                    if np.any(trigger[-int(len(trigger)/3):]<=1.5):    # If final third of signal falls below 1.5 
                        short_triggers.append(trigger)
                    else:
                        clean_triggers.append(trigger) 

            self.clean_triggers[key] = np.array(clean_triggers)
            self.noisy_triggers[key] = np.array(noisy_triggers)
            self.short_triggers[key] = np.array(short_triggers)

    def print_findings(self):
        for key in self.data:
            print(f"\nUSER {key}")
            print(f"Signal shape: {self.data[key].squeeze().shape}")
            print(f"Excluded noisy triggers: {self.noisy_triggers[key].shape}")
            print(f"Excluded short triggers: {self.short_triggers[key].shape}")
            print(f"Included clean triggers: {self.clean_triggers[key].shape}")

            ratio = (len(self.short_triggers[key]) + len(self.noisy_triggers[key])) / len(self.clean_triggers[key])
            print(f"Fraction of excluded/included triggers:{ratio:.2f}")

    def plot_signal(self, key):
        plot_flatten(self.data[key])    # Created sample for figures [53000:55000]

    def plot_clean(self, key):
        plot_grid(self.clean_triggers[key])

    def plot_discarded(self, key):
        plot_flatten(self.noisy_triggers[key])
        plot_flatten(self.short_triggers[key])

    def plot_means_std(self):
        plot_X(*dict_to_X_y(self.clean_triggers))

    def get_triggers(self):
        return self.clean_triggers

    def save_dict(self, save_path):
        data_to_save = {}
        for key in self.data:
            data_to_save[key+"_triggers"] = self.clean_triggers[key] 

        np.savez(save_path, **data_to_save)
        print("\n\n")
        for k in data_to_save: print(f"Saved {k} : {data_to_save[k].shape}")
        print(f"\nSaved file in {str(save_path)}")


def dict_to_X_y(dict):
    """
    Dirty function to convert data from dict format to X, y
    Only used to input correct format on plotting function. 
    """
    X = []
    y = []
    for i, key in enumerate(dict):
        X.append(dict[key])
        y.append(np.full(len(dict[key]), i))
    return np.concatenate(X)[:, np.newaxis, :], np.concatenate(y)



