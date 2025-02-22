import pandas as pd
from criteria import extract_features

"""
Loads raw dataset, extracts features using function made in criteria.py, and saves processed dataset
@param input_fp - Path to the input CSV (raw dataset)
@param output_fp - Path to save the processed CSV.

"""
def process_dataset (input_fp, output_fp):
    df = pd.read_csv(input_fp)
    
    #rename generated label to Label
    #dependent on csv file format
    df.rename(columns={"generated": "Label"}, inplace = True)

    feature_list = []
    for text in df["Text"]:
        features = extract_featuress(text)
        feature_list.append(features)
    
    feature_dicts = []
    for feature in feature_list:
        feature_dicts.append(feature)
    
    feature_df = pd.DataFrame(feature_dicts)

    labels = []
    for label in df["Label"]:
        labels.append(label)
    
    feature_df["Label"] = labels

    feature_df.to_csv(output_fp, index = False)
    