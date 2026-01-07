import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
import sys
import os 
import joblib
import mlflow

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.fe_transformers import (
    GeneRiskEstimator, AddAcmgRules, OriginRareLabelEncoder, 
    ZeroImputer, VariantsAtTributeExtractor, ImpactScoreEncoder
)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
mlruns_path = os.path.join(project_root, 'mlruns')
MLFLOW_URI = f"file:///{mlruns_path.replace(os.sep, '/')}"
BEST_RUN_ID = '476d00e0618146548af06cc7edf67687'

def read_params_from_mflow(run_id):
    mlflow.set_tracking_uri(MLFLOW_URI)
    run = mlflow.get_run(run_id)
    params = run.data.params
    converted_params = {
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth']),
        'learning_rate': float(params['learning_rate']),
        'subsample': float(params['subsample']),
        'colsample_bytree': float(params['colsample_bytree']),
        'gamma': float(params.get('gamma', 0)),
        'min_child_weight': int(params.get('min_child_weight', 1)),
        'reg_alpha': float(params.get('reg_alpha', 0)),
        'reg_lambda': float(params.get('reg_lambda', 1)),
        'objective': 'binary:logistic',
        'n_jobs': -1,
        'random_state': 42
    }
    return converted_params

def train_and_save():
    base_dir = os.path.dirname(__file__)
    X_train_path = os.path.join(base_dir, 'data', 'X_train.csv')
    y_train_path = os.path.join(base_dir,'data', 'y_train.csv')
    model_output_path = os.path.join(base_dir, '..', 'model.pkl')
    X = pd.read_csv(X_train_path)
    y = pd.read_csv(y_train_path)
    cols_to_ohe = ['ORIGIN_GROUPED', 'CHROM']
    cols_to_scale = ['POS', 'DIFF_LEN']

    preprocessor = Pipeline([
        ('zero_imputer', ZeroImputer()),
        ('variants_extractor', VariantsAtTributeExtractor()),
        ('impact_encoder', ImpactScoreEncoder()),
        ('gene_risk', GeneRiskEstimator(m=10)),
        ('origin_grouped', OriginRareLabelEncoder(threshold=50)),
        ('ohe_step', ColumnTransformer(
            [
                ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cols_to_ohe)
            ], 
            remainder='passthrough',
            verbose_feature_names_out=False
        ).set_output(transform='pandas')),
        ('acmg_rules', AddAcmgRules()),
        ('scaler_step', ColumnTransformer([
            ('scaler', StandardScaler(), cols_to_scale)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
        ).set_output(transform='pandas'))
    ])
    best_params = read_params_from_mflow(BEST_RUN_ID)
    model = XGBClassifier(**best_params)
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    final_pipeline.fit(X, y)
    joblib.dump(final_pipeline, model_output_path)

if __name__ == '__main__':
    train_and_save()
