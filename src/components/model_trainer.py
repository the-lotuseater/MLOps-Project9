import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj
from sklearn.model_selection import GridSearchCV

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('Splitting test and train input data')
            X_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            X_test,y_test = test_arr[:,:-1], test_arr[:,-1]
            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'XGBRegressor': XGBRegressor(),
                'CatBoosting Regressor': CatBoostRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor()
            }
            hyper_params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            model_report: dict = self.evaluate_model(X_train,X_test,y_train,y_test,models,hyper_params)
            best_model_name = max(model_report,key=model_report.get)
            best_model_score = model_report[best_model_name]
            if best_model_score<0.6:
                raise CustomException('ERROR: No model found that can fit the data well.')
            best_model = models[best_model_name]
            save_obj(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            logging.info('Saved best model...')
            logging.info('EXIT')
            return (
                    train_arr,
                    test_arr,
                    best_model_score
                )
        except Exception as e:
            raise CustomException(e,sys)
    
    def evaluate_model(self,X_train, X_test, y_train, y_test,models,hyper_params):
        try:
            report = {}
            for model_name,model in models.items():
                params = hyper_params[model_name]
                gs = GridSearchCV(model,params,cv=3)
                gs.fit(X_train,y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train,y_train)
                
                y_test_pred = model.predict(X_test)
                test_model_score = r2_score(y_test,y_test_pred)
                report[model_name] = test_model_score
            return report
        except Exception as e:
            raise Exception(e,sys)