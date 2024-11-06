'''program to split dataset to training and validation set'''
import sys
import random
import pandas as pd
import numpy as np

def randomized_selection(arr:list, seed:int=42) -> list:
    '''Seed to allow same randomisation when sampling. Default of 42 if no seed specified'''
    randomized_list:list = []
    random.seed(seed)
    while len(arr) > 0:
        index = random.randint(0, len(arr) - 1)
        randomized_list.append(arr.pop(index))
    return randomized_list

def stratify(df:pd.DataFrame) -> tuple[list]:
    '''function to stratify dataset as per ratio of classes so split samples are not biased.
    80:20 split between training/validation'''
    validation_m_count:int = int(len(df[df[1] =="M"]) * 0.2)
    validation_b_count:int = int(len(df[df[1] == "B"]) * 0.2)
    converted_list:list = df.to_numpy().tolist()
    validation_set:list = []
    training_set:list = []
    
    for _, value in enumerate(converted_list):
        if value[1] == "M" and validation_m_count > 0:
            validation_set.append(value)
            validation_m_count -= 1
        elif value[1] == "B" and validation_b_count > 0:
            validation_set.append(value)
            validation_b_count -= 1
        else:
            training_set.append(value)
    return (training_set, validation_set)


def main()->None:
    '''shuffle, split and stratify data to training_set_csv and validation_set.csv'''
    try:
        df = pd.read_csv(sys.argv[1], header=None)
    except Exception as e:
        print(f"Error:{e}")
    if (len(sys.argv) == 3):
        randomised_df:pd.DataFrame = pd.DataFrame(randomized_selection(df.values.tolist(), int(sys.argv[2])))
    else:
        randomised_df:pd.DataFrame = pd.DataFrame(randomized_selection(df.values.tolist()))
    training_set, validation_set = stratify(randomised_df)
    pd.DataFrame(training_set).to_csv("data_train.csv", index=False, header=None)
    pd.DataFrame(validation_set).to_csv("data_test.csv", index=False, header=None)


if __name__ == "__main__":
    main()