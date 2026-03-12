import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocess
from perceptrons import SLP, adaline
import random


def main():
    df = pd.read_csv("penguins.csv", na_values=["NA"])

    score = 0
    classes=["Adelie", "Chinstrap", "Gentoo"]
    features=["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass", "OriginLocation"]
    model = -1
    tries = 250
    for _ in range(tries):
        class_pairs = get_random_pairs(classes)
        feature_pairs = get_random_pairs(features)
        model = random.getrandbits(1)

        print("Class pairs:", class_pairs)
        print("Feature pairs:", feature_pairs)
        print("Model:", "SLP" if model == 0 else "Adaline")
        train_df, test_df = preprocess(df, classes=class_pairs, features=feature_pairs, mlp=False)
        # print(train_df)
        # print(test_df)
        # train_df.to_csv("preprocessed_penguins_train.csv", index=False)
        # test_df.to_csv("preprocessed_penguins_test.csv", index=False)
        if model == 0:
            use_bias = random.getrandbits(1)
            print("Using bias:", use_bias)
            weights, bias, acc, y_test, y_pred, errors = SLP(train_df, test_df, use_bias=use_bias)
            score += acc
        else:
            weights, bias, acc, y_test, y_pred, errors = adaline(train_df, test_df)
            score += acc
    print("Average accuracy over 250 runs:", (score/tries)*100, "%")


def get_random_pairs(original_list):
    temp_list = original_list[:]
    random.shuffle(temp_list)

    new_list_of_pairs = []
    if 1 < len(temp_list):
        new_list_of_pairs.append(temp_list[0])
        new_list_of_pairs.append(temp_list[1])

    return new_list_of_pairs


if __name__ == "__main__":
    main()