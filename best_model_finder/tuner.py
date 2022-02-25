from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics  import roc_auc_score,accuracy_score

class Model_Finder:
    """
                This class shall  be used to find the model with best accuracy and AUC score.
                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """

    def __init__(self, file_object, logger_object):
        self.file_object= file_object
        self.logger_object= logger_object


    def get_best_params_for_svc(self,X_train,y_train):
        """
                Method Name: get_best_params_for_svc
                Description: get the parameters for Support Vector Classifier Algorithm which give the best accuracy.
                             Use Hyper Parameter Tuning.
                Output: The model with the best parameters
                On Failure: Raise Exception

                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_svc method of the Model_Finder class')

        try:
            # Base model creation
            svc= SVC()

            # Initializing with different combination of hyperparameters
            param_grid = {"C":[0.001, 0.01, 0.1, 0.5, 1],
                               "kernel": ['linear', 'poly', 'rbf'],
                               'degree': [2, 3], 'gamma':['scale', 0.1, 0.2]}

            # Creating an object of the GridSearchCV
            grid = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5,  verbose=3)

            # Training the GridSearchCV model
            grid.fit(X_train,y_train)

            self.logger_object.log(self.file_object,
                                   'Support Vector best params: '+str(grid.best_params_)+'. Exited the get_best_params_for_svc method of the Model_Finder class')

            # Returning the best estimator in the GridSearchCV model
            return grid.best_estimator_

        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured in get_best_params_for_svc method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Support Vector Classifier Parameter tuning  failed. Exited the get_best_params_for_svc method of the Model_Finder class')
            raise Exception()



    def get_best_params_for_dtree(self,X_train,y_train):
        """
                Method Name: get_best_params_for_dtree
                Description: get the parameters for Decision Tree Algorithm which give the best accuracy.
                             Use Hyper Parameter Tuning.
                Output: The model with the best parameters
                On Failure: Raise Exception

                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_dtree method of the Model_Finder class')

        try:
            # Base model creation
            dtree= DecisionTreeClassifier()

            # Initializing with different combination of hyperparameters
            param_grid = {"criterion": ['gini', 'entropy'],
                                "max_depth": [3,5,7,None], "class_weight": ['balanced', None]}

            # Creating an object of the GridSearchCV
            grid = GridSearchCV(estimator=dtree, param_grid=param_grid, cv=5,  verbose=3)

            # Training the GridSearchCV model
            grid.fit(X_train,y_train)

            self.logger_object.log(self.file_object,
                                   'Decision Tree best params: '+str(grid.best_params_)+'. Exited the get_best_params_for_dtree method of the Model_Finder class')

            # Returning the best estimator in the GridSearchCV model
            return grid.best_estimator_

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_dtree method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'Decision Tree Parameter tuning  failed. Exited the get_best_params_for_dtree method of the Model_Finder class')
            raise Exception()



    def get_best_params_for_random_forest(self,X_train,y_train):
        """
                Method Name: get_best_params_for_random_forest
                Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                             Use Hyper Parameter Tuning.
                Output: The model with the best parameters
                On Failure: Raise Exception

                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_random_forest method of the Model_Finder class')

        try:
            clf= RandomForestClassifier()

            # Initializing with different combination of hyperparameters
            param_grid = {"n_estimators": [50, 100], "criterion": ['gini', 'entropy'],
                               "max_depth": [2,3,5], "max_features": ['auto', 'log2']}

            # Creating an object of the GridSearchCV
            grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5,  verbose=3)

            # Training the GridSearchCV model
            grid.fit(X_train, y_train)

            self.logger_object.log(self.file_object,
                                   'Random Forest best params: '+str(grid.best_params_)+'. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            # Returning the best estimator in the GridSearchCV model
            return grid.best_estimator_

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()


    def get_best_params_for_xgboost(self,X_train,y_train):

        """
                Method Name: get_best_params_for_xgboost
                Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                             Use Hyper Parameter Tuning.
                Output: The model with the best parameters
                On Failure: Raise Exception

                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """
        self.logger_object.log(self.file_object,'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            xgb= XGBClassifier(objective='binary:logistic')

            # Initializing with different combination of hyperparameters
            param_grid = {'learning_rate': [0.5, 0.1, 0.01],
                                       'max_depth': [3, 5, 7],'n_estimators': [50, 100]}

            # Creating an object of the GridSearchCV
            grid= GridSearchCV(xgb, param_grid, verbose=3, cv=5)

            # Training the GridSearchCV model
            grid.fit(X_train, y_train)

            self.logger_object.log(self.file_object,
                                   'XGBoost best params: ' + str(grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')

            # Returning the best estimator in the GridSearchCV model
            return grid.best_estimator_

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()


    def get_best_model(self,X_train,y_train,X_test,y_test):
        """
                Method Name: get_best_model
                Description: Find out the Model which has the best AUC score.
                Output: The best model name and the model object
                On Failure: Raise Exception

                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')

        try:
            # Create the best model for Support Vector Classifier (support_vector; in previous method=svc; get_best_params_for_svc)
            support_vector=self.get_best_params_for_svc(X_train,y_train)
            prediction_support_vector= support_vector.predict(X_test) # prediction using the Support Vector Classifier Algorithm

            if len(y_test.unique()) == 1: # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                support_vector_score = accuracy_score(y_test, prediction_support_vector)
                self.logger_object.log(self.file_object, 'Accuracy for SVC:' + str(support_vector_score))
            else:
                support_vector_score = roc_auc_score(y_test, prediction_support_vector) # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for SVC:' + str(support_vector_score))


            # Create the best model for Decision Tree (dt; in previous method=dtree; get_best_params_for_dtree)
            dt=self.get_best_params_for_dtree(X_train,y_train)
            prediction_dt= dt.predict(X_test) # prediction using the Decision Tree Algorithm

            if len(y_test.unique()) == 1: # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                dt_score = accuracy_score(y_test, prediction_dt)
                self.logger_object.log(self.file_object, 'Accuracy for DT:' + str(dt_score))
            else:
                dt_score = roc_auc_score(y_test, prediction_dt) # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for DT:' + str(dt_score))

            # Create the best model for Random Forest (random_forest; in previous method=clf)
            random_forest=self.get_best_params_for_random_forest(X_train,y_train)
            prediction_random_forest=random_forest.predict(X_test) # prediction using the Random Forest Algorithm

            if len(y_test.unique()) == 1: # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                random_forest_score = accuracy_score(y_test, prediction_random_forest)
                self.logger_object.log(self.file_object, 'Accuracy for RF:' + str(random_forest_score))
            else:
                random_forest_score = roc_auc_score(y_test, prediction_random_forest) # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for RF:' + str(random_forest_score))

            # Create the best model for XGBoost (xgboost; in previous method=xgb)
            xgboost= self.get_best_params_for_xgboost(X_train,y_train)
            prediction_xgboost = xgboost.predict(X_test) # Predictions using the XGBoost Model

            if len(y_test.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                xgboost_score = accuracy_score(y_test, prediction_xgboost)
                self.logger_object.log(self.file_object, 'Accuracy for XGBoost:' + str(xgboost_score))  # Log AUC
            else:
                xgboost_score = roc_auc_score(y_test, prediction_xgboost) # AUC for XGBoost
                self.logger_object.log(self.file_object, 'AUC for XGBoost:' + str(xgboost_score)) # Log AUC


            # Finding the model with highest score

            scores = [support_vector_score, dt_score, random_forest_score, xgboost_score]
            model_idx = scores.index(max(scores))
            if model_idx==0:
                return "SupportVector", support_vector
            elif model_idx==1:
                return "DecisionTree", dt
            elif model_idx==2:
                return 'RandomForest', random_forest
            else:
                return 'XGBoost', xgboost

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()
