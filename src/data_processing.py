
import pandas as pd
from criteria import extract_features


def process_dataset (input_fp, output_fp):
    df = pd.read_csv(input_fp)
    

    df.rename(columns={"text": "Text", "generated": "Label"}, inplace = True)

    feature_list = []
    for text in df["Text"]:
        features = extract_features(text)
        feature_list.append(features)
    

    feature_df = pd.DataFrame(feature_list)

    feature_df["Label"] = df["Label"]



    feature_df.to_csv(output_fp, index = False)

