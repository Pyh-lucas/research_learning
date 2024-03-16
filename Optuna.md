[EDA + Otuna Ensemble (lgb, xgb, cat) | Kaggle](https://www.kaggle.com/code/shadechen/eda-otuna-ensemble-lgb-xgb-cat)
[💸Bank Churn | 💵 kFold | LGBM+Cat+XGB Ensemble 🚀 | Kaggle](https://www.kaggle.com/code/iqmansingh/bank-churn-kfold-lgbm-cat-xgb-ensemble)


其中一个model的代码调参过程

```python
#since the LGBM model is relatively better, Let's try tunning The lgbmodel via Optuna
import optuna
def objective(trial):
    params = {
        'n_estimators' : trial.suggest_int('n_estimators',50,500),
        "max_depth":trial.suggest_int('max_depth',3,50),
        "learning_rate" : trial.suggest_float('learning_rate',1e-4, 0.25, log=True),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        "min_child_weight" : trial.suggest_float('min_child_weight', 0.5,4),
        "min_child_samples" : trial.suggest_int('min_child_samples',1,100),
        "subsample" : trial.suggest_float('subsample', 0.4, 1),
        "subsample_freq" : trial.suggest_int('subsample_freq',0,5),
        "colsample_bytree" : trial.suggest_float('colsample_bytree',0.2,1),
        'num_leaves' : trial.suggest_int('num_leaves', 2, 64),
    }
    lgbmmodel_optuna = LGBMClassifier(**params, device = 'gpu', random_state=42)
    cv = abs(cross_val_score(lgbmmodel_optuna, X, y, cv = 4,scoring='neg_log_loss').mean())
    return cv
```

```python
%%time
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, timeout=2000)
```

```python
#the best value: 0.4341508071086889, I've run the study locally
lgb_params = {'n_estimators': 479,
 'max_depth': 25,
 'learning_rate': 0.011780201673685327,
 'reg_alpha': 0.010913316411519302,
 'reg_lambda': 0.024314296610050013,
 'min_child_weight': 3.100349976889409,
 'min_child_samples': 99,
 'subsample': 0.4975668919008305,
 'subsample_freq': 0,
 'colsample_bytree': 0.3960164652068705,
 'num_leaves': 63}

lgb_opt = LGBMClassifier(**lgb_params, random_state=42, 
                         #device = 'gpu'
                        )
lgb_opt.fit(X, y)
preds = lgb_opt.predict_proba(test_engineered)
preds = pd.DataFrame(preds, columns = ['C', 'CL', 'D'])
```


ensemble3个model

```python
###################BUG FOUND:  JUPYTER WILL NOT RUN IF YOU TRY TO SET THE PARAMS ABOVE AS GPU TRAINER###################
Ensemble = VotingClassifier(estimators = [('lgb', lgb_opt), ('xgb', xgb_opt), ('CB', CB_opt)], 
                            voting='soft',
                            weights = [0.4,0.6,0.0]   #Adjust weighting since XGB performs better in local environment
                            )
Ensemble.fit(X, y)
preds = Ensemble.predict_proba(test_engineered)
preds = pd.DataFrame(preds, columns = ['C', 'CL', 'D'])
```