import pandas as pd


# Split into test and train data
#
# Takes dataframe as input df and a string as input horse
def split_by_horse(df, horse):
    test = df[df['horse'].str.match(horse)]
    train = df.drop(df[df['horse'].str.match(horse)].index)
    test = test.reset_index(drop=True)
    train = train.reset_index(drop=True)
    return test, train
