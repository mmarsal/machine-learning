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
        # Split the training data into two groups (disease/no disease)
        grouped = data.groupby(target_name)
        no_disease_group = grouped.get_group(False)
        disease_group = grouped.get_group(True)

        # Calculate prior probabilities
        total_count = len(data)
        self.priors['disease'] = len(disease_group) / total_count
        self.priors['no_disease'] = len(no_disease_group) / total_count

        # Drop label column and setup parameters
        features = data.columns.drop(target_name)
        self.likelihoods = {'disease': {}, 'no_disease': {}}
        self.gaussian_parameters = {'disease': {}, 'no_disease': {}}

        # Calculate likelihoods and gaussian parameters
        for i, feature in enumerate(features):
            if self.continuous[i]:  # If continuous feature
                self.gaussian_parameters['disease'][feature] = {
                    'mean': disease_group[feature].mean(),
                    'std': disease_group[feature].std()
                }
                self.gaussian_parameters['no_disease'][feature] = {
                    'mean': no_disease_group[feature].mean(),
                    'std': no_disease_group[feature].std()
                }
            else:  # If discrete feature
                self.likelihoods['disease'][feature] = disease_group[feature].value_counts(normalize=True).to_dict()
                self.likelihoods['no_disease'][feature] = no_disease_group[feature].value_counts(
                    normalize=True).to_dict()

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
            # Convert to log to avoid underflow
            log_probability_disease = math.log(self.priors['disease'])
            log_probability_no_disease = math.log(self.priors['no_disease'])

            # Calculate probabilities
            for i, feature in enumerate(row.index):
                feature_value = row[feature]

                if self.continuous[i]:  # If continuous feature
                    # Use help function to calculate Gaussian likelihood
                    mean_disease = self.gaussian_parameters['disease'][feature]['mean']
                    std_disease = self.gaussian_parameters['disease'][feature]['std']
                    likelihood_disease = self.calculate_gaussian_likelihood(feature_value, mean_disease, std_disease)

                    mean_no_disease = self.gaussian_parameters['no_disease'][feature]['mean']
                    std_no_disease = self.gaussian_parameters['no_disease'][feature]['std']
                    likelihood_no_disease = self.calculate_gaussian_likelihood(feature_value, mean_no_disease,
                                                                               std_no_disease)

                    # Add smoothing to avoid probabilities being 0
                    log_probability_disease += math.log(likelihood_disease + 1e-6)
                    log_probability_no_disease += math.log(likelihood_no_disease + 1e-6)
                else:  # If discrete feature
                    # Use already stored probabilities and add smoothing again
                    likelihood_disease = self.likelihoods['disease'].get(feature, {}).get(feature_value, 1e-6)
                    likelihood_no_disease = self.likelihoods['no_disease'].get(feature, {}).get(feature_value, 1e-6)

                    log_probability_disease += math.log(likelihood_disease + 1e-6)
                    log_probability_no_disease += math.log(likelihood_no_disease + 1e-6)

            # Convert back to normal probabilities
            probability_disease = math.exp(log_probability_disease)
            probability_no_disease = math.exp(log_probability_no_disease)

            # Ensure they sum up to 1 and calculate proportion
            total_probabilities = probability_disease + probability_no_disease
            probability_disease /= total_probabilities
            probability_no_disease /= total_probabilities

            # Store the probabilities and final classification
            probabilities.append({
                'P(disease)': probability_disease,
                'P(no_disease)': probability_no_disease,
                'Prediction': True if probability_disease > probability_no_disease else False
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
        predictions = results.iloc[:, -1]

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

        return accuracy, {"true_positives": true_positives, "true_negatives": true_negatives,
                          "false_positives": false_positives, "false_negatives": false_negatives}
