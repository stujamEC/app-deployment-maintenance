import unittest
import os
import sys
import pickle
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


sys.path.append("..")
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)


from process_data import pre_process_data


dir_name = os.path.dirname(__file__)
model_path = os.path.abspath(os.path.join(dir_name, "../webservice/static/credit_model.pkl"))
dict_vectoriser_path = os.path.abspath(os.path.join(dir_name, "../webservice/static/dict_vectorizer.pkl"))
data_scaler_path = os.path.abspath(os.path.join(dir_name, "../webservice/static/data_scaler.pkl"))
test_data_path = os.path.abspath(os.path.join(dir_name, "../german_credit_test.csv"))


class TestModelPerformance(unittest.TestCase):
    """

    """
    def setUp(self):
        if 'credit_model' not in globals():
            with open(model_path, 'rb') as stream:
                self.credit_model = pickle.load(stream)
            # load the DictVectoriser
            with open(dict_vectoriser_path, 'rb') as stream:
                self.dict_vectoriser = pickle.load(stream)
            with open(data_scaler_path, 'rb') as stream:
                self.data_scaler = pickle.load(stream)

    def test_model_precision_recall(self):
        """
        Test if the model's precision and recall on the test set are respectively higher than
        80% and 60%
        """
        # Preparing test data
        test_data = pd.read_csv(test_data_path)
        X_test = pre_process_data(test_data)
        X_test = X_test.drop(columns=['Age (years)'])
        dummy_test = self.dict_vectoriser.transform(X_test.to_dict('records'))
        cols = self.dict_vectoriser.get_feature_names()
        # Convert the data back to data frame
        dummy_test = pd.DataFrame(dummy_test, index=X_test.index, columns=cols)
        new_cols = {}
        for key in cols:
            new_cols[key] = key.replace('=', '_')
        dummy_test = dummy_test.rename(columns=new_cols)
        # Getting the predictions from the test data
        y_preds = self.credit_model.predict(self.data_scaler.transform(dummy_test))
        y_test = test_data['Creditability']
        recall = recall_score(y_test, y_preds, average='binary')
        precision = precision_score(y_test, y_preds, average='binary')

        self.assertTrue(precision*100 > 80)
        self.assertTrue(recall*100 > 60)





