
# Functions used for processing raw data
# i.e. Selecting triggers and releases and plots

import numpy as np
import matplotlib.pyplot as plt

class TriggersAndReleases:

    def __init__(self, signal : np.array):
        self.signal = signal.flatten()

    def run(self):
        self.find_idxs()
        self.cut_windows()
        self.print_findings()
        

    def find_idxs(self, zero_threshold=0.05):
        """
        Finds the indexes of the signal for the beggining of trigger and end of release
        """
        zerosIdx = np.argwhere(self.signal<=zero_threshold)        # Define zero as anything below harcoded value 
        jumpIdx = np.argwhere(np.diff(zerosIdx[:, 0])>1)   # Non-consecutive zeros
        trigIdx =  zerosIdx[jumpIdx]
        releaseIdx = zerosIdx[jumpIdx+1]

        self.trigger_idxs = trigIdx.flatten()
        self.release_idxs = releaseIdx.flatten()


    def cut_windows(self, width=32):
        clean_triggers = []
        clean_releases = []
        noisy_triggers = []
        short_triggers = []

        for i, j in zip(self.trigger_idxs, self.release_idxs):
            trigger = self.signal[i:i+width]
            release = self.signal[j-width:j]

            if np.mean(trigger<=0.5)>=0.8:   # If 80% of the signal is below 0.5 
                noisy_triggers.append(trigger)
            else:
                if np.any(trigger[-int(len(trigger)/3):]<=1.5):    # If final third of signal does not fall below 1.5 
                    short_triggers.append(trigger)
                else:
                    # TODO: Need to confirm that approach below is matching up triggers to releases correctly
                    clean_triggers.append(trigger) 
                    clean_releases.append(release)

        self.clean_triggers = np.array(clean_triggers)
        self.clean_releases= np.array(clean_releases)
        self.noisy_triggers = np.array(noisy_triggers)
        self.short_triggers = np.array(short_triggers)

    def print_findings(self):
        print(f"\nSignal shape: {self.signal.shape}")
        print(f"Excluded noisy triggers: {self.noisy_triggers.shape}")
        print(f"Excluded short triggers: {self.short_triggers.shape}")
        print(f"Included clean triggers: {self.clean_triggers.shape}")
        print(f"Included clean releases: {self.clean_releases.shape}")

    def plot_signal(self, upTo=5000):
        plt.figure(figsize=(30, 5))
        signal = self.signal[:upTo]
        plt.plot(range(len(signal)), signal, "b.")

    def plot_clean(self):
        plot(self.clean_triggers)
        plot(self.clean_releases)
        # TODO: Change to plot_concat and find an interactive way to look at data on notebook

    def plot_noisy(self):
        plot_concat(self.noisy_triggers)

    def plot_short(self):
        plot_concat(self.short_triggers)

    def get_triggers(self):
        return self.clean_triggers

    def get_releases(self):
        return self.clean_releases

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
