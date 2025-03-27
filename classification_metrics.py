def build_confusion_matrix(actual, predicted, labels=None):
    """
    Build a confusion matrix for multi-class classification.
    
    :param actual: List of ground-truth labels
    :param predicted: List of predicted labels
    :param labels: (Optional) A list of all class labels in desired order.
                   If None, they will be inferred from the data.
    :return: (conf_matrix, label_list)
        conf_matrix: 2D list (matrix) where conf_matrix[i][j] is the count
                     of samples with actual label i and predicted label j.
        label_list: The list of labels in the order they appear in the matrix.
    """
    # If no label set is provided, infer from all unique values in actual + predicted
    if labels is None:
        labels = sorted(set(actual) | set(predicted))
    
    # Create a dictionary to map each label to a matrix index
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    # Initialize confusion matrix
    n_labels = len(labels)
    conf_matrix = [[0] * n_labels for _ in range(n_labels)]
    
    # Populate confusion matrix
    for a, p in zip(actual, predicted):
        i = label_to_idx[a]
        j = label_to_idx[p]
        conf_matrix[i][j] += 1
    
    return conf_matrix, labels


def accuracy_score(actual, predicted):
    """
    Calculate overall accuracy: (number correct) / (total samples).
    """
    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    return correct / len(actual)


def precision_recall_f1(conf_matrix, labels):
    """
    Given a confusion matrix and the corresponding label list,
    compute precision, recall, and F1 for each label (macro-averaged).
    
    :param conf_matrix: 2D list (matrix), conf_matrix[i][j] = #samples 
                        with actual=labels[i], predicted=labels[j]
    :param labels: List of labels in the order of matrix indices
    :return: (precision_dict, recall_dict, f1_dict, macro_precision, macro_recall, macro_f1)
    """
    n_labels = len(labels)
    
    # Initialize dictionaries to store metrics per label
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}
    
    for i, label in enumerate(labels):
        # True positives: diagonal
        tp = conf_matrix[i][i]
        
        # Predicted positives: sum of column i
        col_sum = sum(conf_matrix[row][i] for row in range(n_labels))
        
        # Actual positives: sum of row i
        row_sum = sum(conf_matrix[i][col] for col in range(n_labels))
        
        # Precision = TP / (TP + FP)
        precision = tp / col_sum if col_sum > 0 else 0.0
        
        # Recall = TP / (TP + FN)
        recall = tp / row_sum if row_sum > 0 else 0.0
        
        # F1 = 2 * (precision * recall) / (precision + recall)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precision_dict[label] = precision
        recall_dict[label] = recall
        f1_dict[label] = f1
    
    # Macro-average (average over labels)
    macro_precision = sum(precision_dict[l] for l in labels) / n_labels
    macro_recall    = sum(recall_dict[l]    for l in labels) / n_labels
    macro_f1        = sum(f1_dict[l]        for l in labels) / n_labels
    
    return (precision_dict, recall_dict, f1_dict,
            macro_precision, macro_recall, macro_f1)


if __name__ == "__main__":
    # EXAMPLE USAGE

    # Suppose these are your actual and predicted labels
    actual_labels =    ["Purple", "Blue", "Blue", "Red", "Red", "Blue", "Blue", "Red",
    "Orange", "Blue", "Blue", "Blue", "Blue", "Gray", "Blue", "Blue",
    "Blue", "Blue", "Blue", "Green", "Blue", "Blue", "Blue", "Blue",
    "Orange", "Orange", "Blue", "Red", "Blue", "Orange", "Blue", "Orange", "Blue",
    "Blue", "Purple", "Gray", "Blue", "Blue", "Blue", "Blue", "Blue",
    "Blue", "Orange", "Blue", "Blue", "Blue", "Red", "Blue", "Blue", "Blue",
    "Blue", "Blue", "Red"]
    predicted_labels = ["Orange", "Blue", "Orange", "Gray", "Gray", "Blue",
    "Blue", "Red", "Orange", "Orange", "Orange", "Orange", "Blue", "Gray",
    "Gray", "Blue", "Blue", "Blue", "Blue", "Yellow",
    "Orange", "Blue", "Blue", "Blue", "Orange", "Orange", "Orange", "Red", "Blue",
    "Yellow", "Blue", "Gray", "Purple", "Blue", "Orange", "Gray",
    "Orange", "Gray", "Blue", "Red", "Blue", "Blue", "Gray",
    "Blue", "Blue", "Gray", "Gray", "Green", "Blue",
    "Blue", "Green", "Red", "Red"]

    # 1. Build the confusion matrix
    conf_matrix, label_list = build_confusion_matrix(actual_labels, predicted_labels)

    # 2. Compute accuracy
    acc = accuracy_score(actual_labels, predicted_labels)

    # 3. Compute precision, recall, F1 (macro-averaged)
    (prec_dict, rec_dict, f1_dict,
     macro_prec, macro_rec, macro_f1) = precision_recall_f1(conf_matrix, label_list)

    # Print results
    print("Model: model-name")
    print("Labels:", label_list)
    print("Confusion Matrix:")
    for row_idx, row in enumerate(conf_matrix):
        print(f"  Actual={label_list[row_idx]:>15}: {row}")
    print(f"\nOverall Accuracy: {acc:.3f}")

    print("\nPer-Class Precision, Recall, F1:")
    for label in label_list:
        print(f"  {label:>15}: "
              f"Precision={prec_dict[label]:.3f}, "
              f"Recall={rec_dict[label]:.3f}, "
              f"F1={f1_dict[label]:.3f}")

    print(f"\nMacro-Averaged Precision: {macro_prec:.3f}")
    print(f"Macro-Averaged Recall:    {macro_rec:.3f}")
    print(f"Macro-Averaged F1:        {macro_f1:.3f}")
