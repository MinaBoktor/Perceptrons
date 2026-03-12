import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def preprocess(df, classes=["Adelie", "Chinstrap", "Gentoo"], features=["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass", "OriginLocation"], mlp=True):

    if not mlp:
        if len(classes) != 2:
            return -1, -1
        elif len(features) != 2:
            return -1, -1

    # Select only the specified features and the target variable "Species"
    df = df[["Species"]+features]

    # Check if "OriginLocation" is in the selected features and handle it accordingly
    if "OriginLocation" in features:
        # Encode the categorical variable "OriginLocation" using one-hot encoding if selected
        df = pd.get_dummies(df, columns=["OriginLocation"], dtype=int)
        features.remove("OriginLocation")

    # Use data from selected classes only
    df = df[df["Species"].isin(classes)]

    # split the data into train and test sets (30 samples for training, 20 for testing)
    train_list = []
    test_list = []

    for cls in df["Species"].unique():
        df_cls = df[df["Species"] == cls].sample(frac=1)
        train_list.append(df_cls.iloc[:30])
        test_list.append(df_cls.iloc[30:])

    # Combine the lists into DataFrames and shuffle the combined sets
    train_df = pd.concat(train_list).sample(frac=1).reset_index(drop=True)
    test_df = pd.concat(test_list).sample(frac=1).reset_index(drop=True)

    # Calculate the median only for numerical columns and fill NAs
    train_df_median = train_df[features].median(numeric_only=True)
    train_df[features] = train_df[features].fillna(train_df_median)
    test_df[features] = test_df[features].fillna(train_df_median)

    # Scale the features using StandardScaler
    columns_to_scale = [col for col in train_df.columns if col != "Species" and not col.startswith("OriginLocation")]
    scaler = StandardScaler()
    train_df[columns_to_scale] = scaler.fit_transform(train_df[columns_to_scale])
    test_df[columns_to_scale] = scaler.transform(test_df[columns_to_scale])

    if not mlp:
        # Encode the target variable "Species" using label encoding
        le = LabelEncoder()
        train_df["Species"] = le.fit_transform(train_df["Species"])
        test_df["Species"] = le.transform(test_df["Species"])
        train_df["Species"] = train_df["Species"].replace({0: -1, 1: 1})
        test_df["Species"] = test_df["Species"].replace({0: -1, 1: 1})
    else:
        # Use data from all classes
        train_df = pd.get_dummies(train_df, columns=["Species"], dtype=int)
        test_df = pd.get_dummies(test_df, columns=["Species"], dtype=int)


    train_df.to_csv("preprocessed_penguins_train.csv", index=False)
    test_df.to_csv("preprocessed_penguins_test.csv", index=False)

    return train_df, test_df
