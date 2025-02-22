
import pandas as pd
from criteria import extract_features

"""
Loads raw dataset, extracts features using function made in criteria.py, and saves processed dataset
@param input_fp - Path to the input CSV (raw dataset)
@param output_fp - Path to save the processed CSV.

"""
# def process_dataset (input_fp, output_fp):
#     df = pd.read_csv(input_fp)
    
#     #rename generated label to Label
#     #dependent on csv file format
#     df.rename(columns={"generated": "Label"}, inplace = True)

#     feature_list = []
#     for text in df["text"]:
#         features = extract_features(text)
#         feature_list.append(features)
    
    # feature_dicts = []
    # for feature in feature_list:
    #     feature_dicts.append(feature)
    
    # feature_df = pd.DataFrame(feature_dicts)

    # labels = []
    # for label in df["generated"]:
    #     labels.append(label)
    
    # feature_df["generated"] = labels

    # feature_df.to_csv(output_fp, index = False)
    
def process_dataset (input_fp, output_fp):
    df = pd.read_csv(input_fp)
    
    #rename generated label to Label
    #dependent on csv file format
    df.rename(columns={"text": "Text", "generated": "Label"}, inplace = True)

    feature_list = []
    for text in df["Text"]:
        features = extract_features(text)
        feature_list.append(features)
    
    # feature_dicts = []
    # for feature in feature_list:
    #     feature_dicts.append(feature)
    
    feature_df = pd.DataFrame(feature_list)

    feature_df["Label"] = df["Label"]

    # labels = []
    # for label in df["Label"]:
    #     labels.append(label)
    
    # feature_df["Label"] = labels

    feature_df.to_csv(output_fp, index = False)


# # Update process_dataset to include debugging
# def process_dataset(input_fp, output_fp):
#     print(f"Reading input file: {input_fp}")
#     df = pd.read_csv(input_fp)
#     print(f"Dataset loaded with {df.shape[0]} rows.")

#     # Rename 'generated' to 'Label' and 'text' to 'Text' if needed
#     print("Renaming columns...")
#     df.rename(columns={"text": "Text", "generated": "Label"}, inplace=True)

#     feature_list = []
#     for i, text in enumerate(df["Text"]):
#         if i % 100 == 0:  # Show progress every 100 rows
#             print(f"Processing row {i + 1}/{df.shape[0]}...")
#         features = extract_features(text)
#         feature_list.append(features)

#     # Create DataFrame for features
#     print("Creating feature DataFrame...")
#     feature_df = pd.DataFrame(feature_list)
#     feature_df["Label"] = df["Label"]

#     # Save processed data to file
#     print(f"Saving processed data to {output_fp}...")
#     feature_df.to_csv(output_fp, index=False)
#     print(f"Processed data saved: {output_fp}")
