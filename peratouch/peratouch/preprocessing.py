# Functions used for processing raw data
# i.e. Selecting triggers and releases and plots

import numpy as np
import matplotlib.pyplot as plt

class TriggersAndReleases:

    def __init__(self, data : dict):
        self.data = data
        self.trigger_idxs = {}
        self.release_idxs = {}
        self.clean_triggers = {}
        self.clean_releases = {}
        self.noisy_triggers = {}
        self.short_triggers = {}

    def run(self):
        self.find_idxs()
        self.cut_windows()
        self.print_findings()
        

    def find_idxs(self, zero_threshold=0.05):
        """
        Finds the indexes of the signal for the beggining of trigger and end of release
        """
        for key in self.data:
            signal = self.data[key].squeeze()     # Read signal of each user

            zerosIdx = np.argwhere(signal<=zero_threshold)        # Define zero as anything below harcoded value 
            jumpIdx = np.argwhere(np.diff(zerosIdx[:, 0])>1)   # Non-consecutive zeros
            trigIdx =  zerosIdx[jumpIdx]
            releaseIdx = zerosIdx[jumpIdx+1]

            self.trigger_idxs[key] = trigIdx.flatten()
            self.release_idxs[key] = releaseIdx.flatten()


    def cut_windows(self, width=32):
        for key in self.data:

            # Initiate empty lists 
            clean_triggers = []
            clean_releases = []
            noisy_triggers = []
            short_triggers = []

            signal = self.data[key].squeeze()

            for i, j in zip(self.trigger_idxs[key], self.release_idxs[key]):
                trigger = signal[i:i+width]
                release = signal[j-width:j]

                if np.mean(trigger<=0.5)>=0.8:   # If 80% of the signal is below 0.5 
                    noisy_triggers.append(trigger)
                else:
                    if np.any(trigger[-int(len(trigger)/3):]<=1.5):    # If final third of signal does not fall below 1.5 
                        short_triggers.append(trigger)
                    else:
                        # TODO: Need to confirm that approach below is matching up triggers to releases correctly
                        clean_triggers.append(trigger) 
                        clean_releases.append(release)

            self.clean_triggers[key] = np.array(clean_triggers)
            self.clean_releases[key] = np.array(clean_releases)
            self.noisy_triggers[key] = np.array(noisy_triggers)
            self.short_triggers[key] = np.array(short_triggers)

    def print_findings(self):
        for key in self.data:
            print(f"\nUSER {key}")
            print(f"Signal shape: {self.data[key].squeeze().shape}")
            print(f"Excluded noisy triggers: {self.noisy_triggers[key].shape}")
            print(f"Excluded short triggers: {self.short_triggers[key].shape}")
            print(f"Included clean triggers: {self.clean_triggers[key].shape}")
            print(f"Included clean releases: {self.clean_releases[key].shape}")

    def plot_signal(self, key, upTo=5000):
        plt.figure(figsize=(30, 5))
        signal = self.data[key].squeeze()[:upTo]
        plt.plot(range(len(signal)), signal, "b.")

    def plot_clean(self, key):
        plot(self.clean_triggers[key])
        plot(self.clean_releases[key])
        # TODO: Change to plot_concat and find an interactive way to look at data on notebook

    def plot_noisy(self, key):
        plot_concat(self.noisy_triggers[key])

    def plot_short(self, key):
        plot_concat(self.short_triggers[key])

    def plot_means_std(self):
        plt.figure(figsize=(13, 7))
        n_users = len(self.data)
        for i, key in enumerate(self.data):
            for j, sig in enumerate([self.clean_triggers[key], self.clean_releases[key]]):
                plt.subplot(n_users, 2, 2*i+j+1)
                sig_avg = np.mean(sig, axis=0)
                sig_std = np.std(sig, axis=0)
                plt.title(key+" mean+std")
                plt.errorbar(np.arange(len(sig_avg)), sig_avg, sig_std, fmt="b.")
                plt.xticks([])

    def get_triggers(self):
        return self.clean_triggers

    def get_releases(self):
        return self.clean_releases

    def save_dict(self, save_path):
        data_to_save = {}
        for key in self.data:
            data_to_save[key+"_triggers"] = self.clean_triggers[key] 
            data_to_save[key+"_releases"] = self.clean_releases[key] 

        np.savez(save_path, **data_to_save)
        print("\n\n")
        for k in data_to_save: print(f"Saved {k} : {data_to_save[k].shape}")
        print(f"\nSaved file in {str(save_path)}")

def plot(triggers, nx_plots=10, ny_plots=2):
    plt.figure(figsize=(nx_plots*2, ny_plots*2))
    for i in range(nx_plots * ny_plots):
        plt.subplot(ny_plots, nx_plots, i+1)
        trig = triggers[i]
        plt.plot(range(len(trig)), trig, "b.")
        plt.xticks([])

def plot_concat(triggers):
    if len(triggers)==0: return 
    false_sig = np.concatenate(triggers)
    plt.figure(figsize=(30, 5))
    plt.plot(np.arange(len(false_sig)), false_sig, "b.")

