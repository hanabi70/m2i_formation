import pandas as pd

def sklearn_to_frame(X,y) -> pd.DataFrame:
    return pd.concat([X, y], axis=1)