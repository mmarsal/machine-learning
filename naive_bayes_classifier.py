import pandas as pd

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
        self.priors = {} # stores prior probabilities
        self.likelihoods = {} # stores likelihoods for discrete features
        self.gaussian_parameters = {} # stores gaussian parameters for continuous features

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
            if self.continuous[i]: # If continuous feature
                self.gaussian_parameters['disease'][feature] = {
                    'mean': disease_group[feature].mean(),
                    'std': disease_group[feature].std()
                }
                self.gaussian_parameters['no_disease'][feature] = {
                    'mean': no_disease_group[feature].mean(),
                    'std': no_disease_group[feature].std()
                }
            else: # If discrete feature
                self.likelihoods['disease'][feature] = disease_group[feature].value_counts(normalize=True).to_dict()
                self.likelihoods['no_disease'][feature] = no_disease_group[feature].value_counts(
                    normalize=True).to_dict()

    def predict_probability(self, data: pd.DataFrame):
        """
        Calculates the Naive Bayes prediction for a whole pd.DataFrame.
        :param data: pd.DataFrame to be predicted
        :return: pd.DataFrame containing probabilities for all categories as well as the classification result
        """
        pass

    def evaluate_on_data(self, data: pd.DataFrame, test_labels):
        """
        Predicts a test DataFrame and compares it to the given test_labels.
        :param data: pd.DataFrame containing the test data
        :param test_labels:
        :return: tuple of overall accuracy and confusion matrix values
        """
        pass
