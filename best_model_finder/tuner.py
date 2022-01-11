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
        self.svc= SVC()
        self.dtree= DecisionTreeClassifier()
        self.clf= RandomForestClassifier()
        self.xgb= XGBClassifier(objective='binary:logistic')


    def get_best_params_for_svc(self,train_x,train_y):
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
            # Initializing with different combination of hyperparameters
            self.param_grid = {"C":[0.001, 0.01, 0.1, 0.5, 1],
                               "kernel": ['linear', 'poly', 'rbf'],
                               'degree': [2, 3], 'gamma':['scale', 0.1, 0.2]}

            # Creating an object of the GridSearchCV
            self.grid = GridSearchCV(estimator=self.svc, param_grid=self.param_grid, cv=5,  verbose=3)

            # Finding the best hyperparameters
            self.grid.fit(train_x, train_y)

            # Extracting the best hyperparameters
            self.C      = self.grid.best_params_['C']
            self.kernel = self.grid.best_params_['kernel']
            self.degree = self.grid.best_params_['degree']
            self.gamma  = self.grid.best_params_['gamma']

            # Creating a new model with the best hyperparameters
            self.svc = SVC(C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma)

            # Training the new model
            self.svc.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Support Vector best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_svc method of the Model_Finder class')

            return self.svc

        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured in get_best_params_for_svc method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Support Vector Classifier Parameter tuning  failed. Exited the get_best_params_for_svc method of the Model_Finder class')
            raise Exception()



    def get_best_params_for_dtree(self,train_x,train_y):
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
            # Initializing with different combination of hyperparameters
            self.param_grid = {"criterion": ['gini', 'entropy'],
                                "max_depth": [3,5,7,None], "class_weight": ['balanced', None]}

            # Creating an object of the GridSearchCV
            self.grid = GridSearchCV(estimator=self.dtree, param_grid=self.param_grid, cv=5,  verbose=3)

            # Finding the best hyperparameters
            self.grid.fit(train_x, train_y)

            # Extracting the best hyperparameters
            self.criterion    = self.grid.best_params_['criterion']
            self.max_depth    = self.grid.best_params_['max_depth']
            self.class_weight = self.grid.best_params_['class_weight']


            # Creating a new model with the best hyperparameters
            self.dtree = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth, class_weight=self.class_weight)

            # Training the new model
            self.dtree.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Decision Tree best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_dtree method of the Model_Finder class')

            return self.dtree

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_dtree method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'Decision Tree Parameter tuning  failed. Exited the get_best_params_for_dtree method of the Model_Finder class')
            raise Exception()



    def get_best_params_for_random_forest(self,train_x,train_y):
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
            # Initializing with different combination of hyperparameters
            self.param_grid = {"n_estimators": [50, 100], "criterion": ['gini', 'entropy'],
                               "max_depth": [2,3,5], "max_features": ['auto', 'log2']}

            # Creating an object of the GridSearchCV
            self.grid = GridSearchCV(estimator=self.clf, param_grid=self.param_grid, cv=5,  verbose=3)

            # Finding the best hyperparameters
            self.grid.fit(train_x, train_y)

            # Extracting the best hyperparameters
            self.n_estimators = self.grid.best_params_['n_estimators']
            self.criterion    = self.grid.best_params_['criterion']
            self.max_depth    = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']

            # Creating a new model with the best hyperparameters
            self.clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                              max_depth=self.max_depth, max_features=self.max_features)
            # Training the new model
            self.clf.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Random Forest best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return self.clf

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()



    def get_best_params_for_xgboost(self,train_x,train_y):

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
            # Initializing with different combination of hyperparameters
            self.param_grid_xgboost = {'learning_rate': [0.5, 0.1, 0.01],
                                       'max_depth': [3, 5, 7],'n_estimators': [50, 100]}

            # Creating an object of the GridSearchCV
            self.grid= GridSearchCV(XGBClassifier(objective='binary:logistic'),self.param_grid_xgboost, verbose=3,cv=5)

            # Finding the best hyperparameters
            self.grid.fit(train_x, train_y)

            # Extracting the best hyperparameters
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth     = self.grid.best_params_['max_depth']
            self.n_estimators  = self.grid.best_params_['n_estimators']

            # Creating a new model with the best hyperparameters
            self.xgb = XGBClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth, n_estimators=self.n_estimators)

            # Training the new model
            self.xgb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'XGBoost best params: ' + str(self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()


    def get_best_model(self,train_x,train_y,test_x,test_y):
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
            self.support_vector=self.get_best_params_for_svc(train_x,train_y)
            self.prediction_support_vector=self.support_vector.predict(test_x) # prediction using the Support Vector Classifier Algorithm

            if len(test_y.unique()) == 1: # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.support_vector_score = accuracy_score(test_y,self.prediction_support_vector)
                self.logger_object.log(self.file_object, 'Accuracy for SVC:' + str(self.support_vector_score))
            else:
                self.support_vector_score = roc_auc_score(test_y, self.prediction_support_vector) # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for SVC:' + str(self.support_vector_score))


            # Create the best model for Decision Tree (dt; in previous method=dtree; get_best_params_for_dtree)
            self.dt=self.get_best_params_for_dtree(train_x,train_y)
            self.prediction_dt=self.dt.predict(test_x) # prediction using the Decision Tree Algorithm

            if len(test_y.unique()) == 1: # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.dt_score = accuracy_score(test_y,self.prediction_dt)
                self.logger_object.log(self.file_object, 'Accuracy for DT:' + str(self.dt_score))
            else:
                self.dt_score = roc_auc_score(test_y, self.prediction_dt) # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for DT:' + str(self.dt_score))

            # Create the best model for Random Forest (random_forest; in previous method=clf)
            self.random_forest=self.get_best_params_for_random_forest(train_x,train_y)
            self.prediction_random_forest=self.random_forest.predict(test_x) # prediction using the Random Forest Algorithm

            if len(test_y.unique()) == 1: # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.random_forest_score = accuracy_score(test_y,self.prediction_random_forest)
                self.logger_object.log(self.file_object, 'Accuracy for RF:' + str(self.random_forest_score))
            else:
                self.random_forest_score = roc_auc_score(test_y, self.prediction_random_forest) # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for RF:' + str(self.random_forest_score))

            # Create the best model for XGBoost (xgboost; in previous method=xgb)
            self.xgboost= self.get_best_params_for_xgboost(train_x,train_y)
            self.prediction_xgboost = self.xgboost.predict(test_x) # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
                self.logger_object.log(self.file_object, 'Accuracy for XGBoost:' + str(self.xgboost_score))  # Log AUC
            else:
                self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost) # AUC for XGBoost
                self.logger_object.log(self.file_object, 'AUC for XGBoost:' + str(self.xgboost_score)) # Log AUC


            # Finding the model with highest score

            self.scores = [self.support_vector_score, self.dt_score, self.random_forest_score, self.xgboost_score]
            self.model_idx = self.scores[self.scores.index(max(self.scores))]
            if self.model_idx==0:
                return "SupportVector", self.support_vector
            elif self.model_idx==1:
                return "DecisionTree", self.dt
            elif self.model_idx==2:
                return 'RandomForest',self.random_forest
            else:
                return 'XGBoost',self.xgboost

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()
