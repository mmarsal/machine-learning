import pandas as pd
import math


class NaiveBayes:
    """
    Naive Bayes classifier for continuous and discrete features using pandas
    """

    def __init__(self, continuous=None):
        """
        :param continuous: list containing a bool for each feature column to be analyzed. True if the feature column
                           contains a continuous feature, False if discrete
        """
        self.continuous = continuous
        self.class_priors = {}
        self.conditional_probs = {}
        self.mean_var = {}

    def fit(self, data: pd.DataFrame, target_name: str):
        """
        Fitting the training data by saving all relevant conditional probabilities for discrete values or for continuous
        features.
        :param data: pd.DataFrame containing training data (including the label column)
        :param target_name: str Name of the label column in data
        """
        self.target_name = target_name
        labels = data[target_name].unique()

        for label in labels:
                label_data = data[data[target_name] == label]
                self.class_priors[label] = len(label_data) / len(data)
                self.conditional_probs[label] = {}
                self.mean_var[label] = {}

                for index, column in enumerate(data.columns.drop(target_name)):
                    if self.continuous[index]:
                         mean = label_data[column].mean()
                         var = label_data[column].var() if len(label_data[column]) > 1 else 1e-6
                         self.mean_var[label][column] = (mean, var)
                    else:
                         probs = label_data[column].value_counts(normalize=True).to_dict()
                         self.conditional_probs[label][column] = probs
    
    def calculateLikelyhood(self, label, column, value, is_continuous):
        if is_continuous:
            # Gausfunktion für continuous features
            mean, var = self.mean_var[label][column]
            
            exponent = math.exp(-(value - mean) ** 2 / (2 * var))
            return (1 / math.sqrt(2 * math.pi * var)) * exponent
        else:
             return self.conditional_probs[label][column].get(value, 1e-6)


    def predict_probability(self, data: pd.DataFrame):
        """
        Calculates the Naive Bayes prediction for a whole pd.DataFrame.
        :param data: pd.DataFrame to be predicted
        :return: pd.DataFrame containing probabilities for all categories as well as the classification result
        """
        results = []

        for _, row in data.iterrows():
            label_probs = {}
            for label in self.class_priors:
                prob = math.log(self.class_priors[label])
                for idx, column in enumerate(data.columns):
                    likelihood = self.calculateLikelyhood(label, column, row[column], self.continuous[idx])
                    prob += math.log(likelihood)
                label_probs[label] = prob
            
            max_log_prob = max(label_probs.values())
            exp_sum = sum(math.exp(prob - max_log_prob) for prob in label_probs.values())
            for label in label_probs:
                label_probs[label] = math.exp(label_probs[label] - max_log_prob) / exp_sum
            
            predicted_label = max(label_probs, key=label_probs.get)
            label_probs['Predicted'] = predicted_label
            results.append(label_probs)

        return pd.DataFrame(results)

    def evaluate_on_data(self, data: pd.DataFrame, test_labels):
        """
        Predicts a test DataFrame and compares it to the given test_labels.
        :param data: pd.DataFrame containing the test data
        :param test_labels:
        :return: tuple of overall accuracy and confusion matrix values
        """
        predictions = self.predict_probability(data)['Predicted']
        accuracy = (predictions.reset_index(drop=True) == test_labels.reset_index(drop=True)).mean()
        
        # Build confusion matrix
        labels = sorted(test_labels.unique())
        confusion_matrix = pd.DataFrame(0, index=labels, columns=labels)
        
        for true, pred in zip(test_labels, predictions):
            confusion_matrix.loc[true, pred] += 1

        return accuracy, confusion_matrix
