
"""Training utilities."""
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

from torchvision import transforms

import importlib
from pathlib import Path

# from sound_classifier.models.hyper_cnn import Net
from sound_classifier.utils.data_utils import normalize_data
from sound_classifier.data.loader import UrbanSound8kDataset
from sound_classifier.transformations.agumentations import MyRightShift, MyAddGaussNoise, MyReshape


def available_models() -> list[str]:
    """Return a list of python modules in sound_classifier.models."""
    here = Path(__file__).parent / "models"
    return [p.stem for p in here.glob("*.py") if p.stem != "__init__"]

def init_model(model_name: str, **build_kwargs):



  if model_name not in available_models():
    raise ValueError(
        f"Model '{model_name}' not found. "
        f"Available: {', '.join(available_models())}"

    )

  module_path = f"sound_classifier.models.{model_name}"
  module = importlib.import_module(module_path)

  if hasattr(module, "build_model"):
      model = module.build_model( **build_kwargs)
  elif hasattr(module, "Model"):
      model_cls = getattr(module, "Model")
      model = model_cls( **build_kwargs)
  else:
      raise AttributeError(
          f"Module '{module_path}' must expose a "
          "`build_model()` function or a `Model` class."
      )

  return model
    # determine if the system supports CUDA
    # if torch.cuda.is_available():
    #   device = torch.device("cuda:0")
    # else:
    #   device = torch.device("cpu")

    # # init model
    # net = Net(device).to(device)

    # return net

def process_fold(fold_k, dataset_df, model_name, epochs=100, batch_size=32, num_of_workers=0):

    # build transformation pipelines for data augmentation
    train_transforms = transforms.Compose([
        # MyRightShift(input_size=128,
        #                                                 width_shift_range=13,
        #                                                 shift_probability=0.9),
        #                                    MyAddGaussNoise(input_size=128,
        #                                                    add_noise_probability=0.55),
                                          MyReshape(output_size=(1,128,128))
                                          ])

    test_transforms = transforms.Compose([MyReshape(output_size=(1,128,128))])

    # split the data
    train_df = dataset_df[dataset_df['fold'] != fold_k]
    test_df = dataset_df[dataset_df['fold'] == fold_k]

    # sampled_df = pd.DataFrame(columns=train_df.columns)

    # for label in train_df['label'].unique():

    #   # Filter the DataFrame to get rows with the current label
    #   df_label = train_df[train_df['label'] == label]

    #   # Sample 10 rows from the filtered DataFrame
    #   sampled_rows = df_label[:1]

    #   # Append the sampled rows to the new DataFrame
    #   sampled_df = pd.concat([sampled_df, sampled_rows])

    # train_df = sampled_df

    # train_df = train_df[:1]

    # normalize the data
    train_df, test_df = normalize_data(train_df, test_df)

    # init train data loader
    train_ds = UrbanSound8kDataset(train_df, transform=train_transforms)
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle = False,
                              # shuffle = True,
                              pin_memory=True,
                              num_workers=num_of_workers)

    # init test data loader
    test_ds = UrbanSound8kDataset(test_df, transform=test_transforms)
    test_loader = DataLoader(test_ds,
                            batch_size=batch_size,
                            shuffle = False,
                            pin_memory=True,
                            num_workers=num_of_workers)

    # init model
    model = init_model(model_name)

    # pre-training accuracy
    score = model.evaluate(test_loader)
    print("Pre-training accuracy: %.4f%%" % (100 * score[1]))

    # train the model
    start_time = datetime.now()
    history = model.fit(train_loader, epochs=epochs, val_loader=test_loader)
    end_time = datetime.now() - start_time
    print("\nTraining completed in time: {}".format(end_time))

    return history
