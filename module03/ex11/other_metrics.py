import numpy as np


def metrics(y,y_hat,  pos_label=1):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(y.shape[0]):
        if y[i] == pos_label and y_hat[i] == pos_label:
            tp += 1
        elif y[i] != pos_label and y_hat[i] == pos_label: 
            fp += 1
        elif y[i] == pos_label and y_hat[i] != pos_label:
            fn += 1
        elif y[i] != pos_label and y_hat[i] != pos_label:
            tn += 1
    return tp, fp, fn, tn
def accuracy_score_(y, y_hat):
    ac = 0
    for yi, y_hati in zip(y, y_hat):
        if yi == y_hati:
            ac += 1
    return ac / len(y)

def precision_score_(y, y_hat, pos_label=1):
    """
	Compute the precision score.
	Args:
	y:a numpy.ndarray for the correct labels
	y_hat:a numpy.ndarray for the predicted labels
	pos_label: str or int, the class on which to report the precision_score (default=1)
	Returns:
	The precision score as a float.
	None on any error.
	Raises:
	This function should not raise any Exception.
	"""
    tp, fp, fn, tn = metrics(y, y_hat, pos_label)
    return (tp / (tp + fp))

def recall_score_(y, y_hat,  pos_label=1):
    """
    Compute the recall score.
	Args:
	y:a numpy.ndarray for the correct labels
	y_hat:a numpy.ndarray for the predicted labels
	pos_label: str or int, the class on which to report the precision_score (default=1)
	Returns:
	The recall score as a float.
	None on any error.
	Raises:
	This function should not raise any Exception.
	"""
    tp, fp, fn, tn = metrics(y, y_hat, pos_label)
    return (tp / (tp + fn))

  
def f1_score_(y, y_hat, pos_label=1):
    """
	Compute the f1 score.
	Args:
	y:a numpy.ndarray for the correct labels
	y_hat:a numpy.ndarray for the predicted labels
	pos_label: str or int, the class on which to report the precision_score (default=1)
	Returns:
	The f1 score as a float.
	None on any error.
	Raises:
	This function should not raise any Exception.
	"""
    ps = precision_score_(y, y_hat, pos_label)
    rc = recall_score_(y, y_hat, pos_label) 
    return ((2 * ps * rc) / (ps + rc))



if __name__ == "__main__":
    y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1])
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0])
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    print(f"My calcu -> accuracy: {accuracy_score_(y, y_hat)}")
    print(f"Library -> accuracy: {accuracy_score(y, y_hat)}")
    print(f"My calcu -> precision: {precision_score_(y, y_hat)}")
    print(f"Library -> precision: {precision_score(y, y_hat)}")
    print(f"My calcu -> recall: {recall_score_(y, y_hat)}")
    print(f"Library -> recall: {recall_score(y, y_hat)}")
    print(f"My calcu -> score: {f1_score_(y, y_hat)}")
    print(f"Library -> score: {score(y, y_hat)}")

    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])
    print(f"accuracy: {accuracy_score_(y, y_hat)}")
    print(f"precision: {precision_score_(y, y_hat, pos_label='dog')}")
    print(f"recall: {recall_score_(y, y_hat, pos_label='dog')}")
    print(f"score: {f1_score_(y, y_hat, pos_label='dog')}")
    
    print(accuracy_score_(y, y_hat))
    print(precision_score_(y, y_hat, pos_label='norminet'))
    print(recall_score_(y, y_hat, pos_label='norminet'))
    print(f1_score_(y, y_hat, pos_label='norminet'))