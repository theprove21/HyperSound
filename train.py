
"""Command-line entry point for training."""
# import argparse
import pandas as pd
from tqdm import tqdm

from sound_classifier.training import init_model, process_fold
# from sound_classifier.utils.data_utils import show_results

import os
import sys
import pickle
import copy
import numpy as np
# from sound_classifier.utils.data_utils import show_results



def main():
    # parser = argparse.ArgumentParser(description="Train the sound classifier.")
    # parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    # args = parser.parse_args()
    dataset_path = "/content/gdrive/MyDrive/US8K/us8k_df.pkl"
    us8k_df = pd.read_pickle(dataset_path)
    FOLD_K = 1
    REPEAT = 1
    model_name = 'cnn_hyp'

    tot_history = []
    tot_labels = []
    for i in range(REPEAT):
        history, labels = process_fold(FOLD_K, us8k_df, model_name=model_name, epochs=100, num_of_workers=4)
        tot_history.append(history)
        tot_labels.append(labels)
    
    print("Training completed.")
        
    all_y_true = []
    all_y_pred = []

    for i in range(len(tot_history)):
        best_epoch = np.argmax(tot_history[i]['val_accuracy'])
        truth = tot_labels[i]['truth'][best_epoch]
        prediction = tot_labels[i]['preds'][best_epoch]
    
        all_y_true.extend(truth)
        all_y_pred.extend(prediction)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    output_dir_history = "/content/hyperbolic_urban_sound_classifier/experiments/cnn_mm_learn_curv/history"

    output_dir_labels = "/content/hyperbolic_urban_sound_classifier/experiments/cnn_mm_learn_curv/labels"

    file_path = os.path.join(output_dir_history, f"history{FOLD_K}.pkl")

    with open(file_path, "wb") as fp:
        pickle.dump(tot_history, fp)


    file_path = os.path.join(output_dir_labels, f"labels{FOLD_K}.pkl")

    with open(file_path, "wb") as fp:
        pickle.dump({"y_true": all_y_true, "y_pred": all_y_pred}, fp)


if __name__ == "__main__":
    main()
