import pandas as pd

def add_features(df):
    df = df.copy()

    df['balls_bowled'] = 120 - df['balls_left']
    df['run_rate'] = df['current_score'] / df['balls_bowled'] * 6
    df['required_run_rate'] = (
        (df['target'] - df['current_score']) / df['balls_left']
    ) * 6

    return df