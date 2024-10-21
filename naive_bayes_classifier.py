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
        self.priors = {}  # stores prior probabilities
        self.likelihoods = {}  # stores likelihoods for discrete features
        self.gaussian_parameters = {}  # stores gaussian parameters for continuous features

    def fit(self, data: pd.DataFrame, target_name: str):
        """
        Fitting the training data by saving all relevant conditional probabilities for discrete values or for continuous
        features.
        :param data: pd.DataFrame containing training data (including the label column)
        :param target_name: str Name of the label column in data
        """
        # Split the training data into groups
        grouped = data.groupby(target_name)

        # Total number of samples
        total_count = len(data)

        # Drop label column and setup features
        features = data.columns.drop(target_name)

        # Calculate prior probabilities
        self.priors = grouped.size().div(total_count).to_dict()

        # Calculate likelihoods and Gaussian parameters
        for i, feature in enumerate(features):
            if self.continuous[i]:  # If continuous feature
                self.gaussian_parameters[feature] = grouped[feature].agg(['mean', 'std']).to_dict(orient='index')
            else:  # If discrete feature
                likelihood = grouped[feature].value_counts(normalize=True).unstack(fill_value=0)
                self.likelihoods[feature] = likelihood.to_dict(orient='index')

    def calculate_gaussian_likelihood(self, feature_value: float, mean: float, std: float):
        """
        Help function to calculate Gaussian likelihood.
        :param feature_value: float current feature being iterated
        :param mean: float The current feature's mean
        :param std: float The current feature's standard deviation
        :return: float Gaussian likelihood
        """
        if std == 0:
            return 0
        exponent = math.exp(-((feature_value - mean) ** 2) / (2 * std ** 2))
        return (1 / (math.sqrt(2 * math.pi) * std)) * exponent

    def predict_probability(self, data: pd.DataFrame):
        """
        Calculates the Naive Bayes prediction for a whole pd.DataFrame.
        :param data: pd.DataFrame to be predicted
        :return: pd.DataFrame containing probabilities for all categories as well as the classification result
        """
        probabilities = []  # Store the final probabilities

        for _, row in data.iterrows():
            log_probabilities = {label: math.log(self.priors[label]) for label in
                                 self.priors}  # Initialize log probabilities for all classes

            # Calculate probabilities
            for i, feature in enumerate(row.index):
                feature_value = row[feature]

                if self.continuous[i]:  # If continuous feature
                    for label in self.priors:
                        mean = self.gaussian_parameters[feature][label]['mean']
                        std = self.gaussian_parameters[feature][label]['std']
                        likelihood = self.calculate_gaussian_likelihood(feature_value, mean, std)
                        # Add smoothing to avoid underflow
                        log_probabilities[label] += math.log(likelihood + 1e-6)
                else:  # If discrete feature
                    for label in self.priors:
                        likelihood = self.likelihoods[feature].get(label, {}).get(feature_value, 1e-6)
                        log_probabilities[label] += math.log(likelihood + 1e-6)

            # Convert back to normal probabilities
            total_probabilities = sum(math.exp(log_prob) for log_prob in log_probabilities.values())
            normalized_probabilities = {label: math.exp(log_prob) / total_probabilities for label, log_prob in
                                        log_probabilities.items()}

            # Store the probabilities and final classification
            prediction = max(normalized_probabilities,
                             key=normalized_probabilities.get)  # Class with the highest probability
            probabilities.append({
                **normalized_probabilities,  # Spread the probabilities into the dictionary
                'Prediction': prediction
            })

        return pd.DataFrame(probabilities)

    def evaluate_on_data(self, data: pd.DataFrame, test_labels: pd.DataFrame):
        """
        Predicts a test DataFrame and compares it to the given test_labels.
        :param data: pd.DataFrame containing the test data
        :param test_labels: pd.DataFrame containing the test labels
        :return: tuple of overall accuracy and confusion matrix values
        """
        # Call predict function and keep the prediction column
        results = self.predict_probability(data)
        predictions = results['Prediction']

        # Initialize confusion matrix
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        # Compare results to the actual labels and calculate confusion matrix values
        for true_label, predicted_label in zip(test_labels, predictions):
            if predicted_label == True and true_label == True:
                true_positives += 1
            elif predicted_label == False and true_label == False:
                true_negatives += 1
            elif predicted_label == True and true_label == False:
                false_positives += 1
            elif predicted_label == False and true_label == True:
                false_negatives += 1

        # Calculate accuracy
        accuracy = (true_positives + true_negatives) / (
                true_positives + true_negatives + false_positives + false_negatives)

        return accuracy, {
            "true_positives": true_positives,
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
