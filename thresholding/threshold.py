import numpy as np

def compute_threshold(errors,method="percentile",value=95):
    if method=="percentile":
        return np.percentile(errors,value)
    elif method=="std":
        return np.mean(errors)+3*np.std(errors)
    elif method=="iqr":
        Q1=np.percentile(errors,25)
        Q3=np.percentile(errors,75)
        IQR=Q3-Q1
        return Q3+1.5*IQR
    else:
        raise ValueError("Unknown threshold method")

def detect(errors,threshold):
    return errors>threshold