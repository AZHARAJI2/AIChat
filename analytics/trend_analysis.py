import pandas as pd

def build_trends(df):
    df["date"] = pd.to_datetime(df["date"])

    trend = (
        df.groupby([df["date"].dt.to_period("M"), "interest"])
        .size()
        .unstack(fill_value=0)
    )

    return trend
