import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
import joblib
import os
import mlflow
import argparse
from sklearn import set_config
set_config(transform_output="pandas")
from fe_transformers import (
    GeneRiskEstimator, AddAcmgRules, OriginRareLabelEncoder, 
    ZeroImputer, VariantsAtTributeExtractor, ImpactScoreEncoder
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--best_run_id', type=str, required=True)
    args = parser.parse_args()
    mlflow.autolog(log_models=False)

    with mlflow.start_run():
        run = mlflow.get_run(args.best_run_id)
        params = run.data.params
        best_params = {
            'n_estimators': int(params['n_estimators']),
            'max_depth': int(params['max_depth']),
            'learning_rate': float(params['learning_rate']),
            'min_child_weight': int(params.get('min_child_weight', 1)),
            'gamma': float(params.get('gamma', 0)),
            'subsample': float(params.get('subsample', 1.0)),
            'colsample_bytree': float(params.get('colsample_bytree', 1.0)),
            'reg_alpha': float(params.get('reg_alpha', 0)),
            'reg_lambda': float(params.get('reg_lambda', 1)),
            'objective': 'binary:logistic',
            'n_jobs': -1,
            'random_state': 42
        }
        X = pd.read_csv(os.path.join(args.data_folder, 'X_train.csv'))
        y = pd.read_csv(os.path.join(args.data_folder, 'y_train.csv'))
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
            )),
            ('acmg_rules', AddAcmgRules()),
            ('scaler_step', ColumnTransformer([
                ('scaler', StandardScaler(), cols_to_scale)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
            ))
        ])
        model = XGBClassifier(**best_params)
        final_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        final_pipeline.fit(X, y)
        os.makedirs('outputs', exist_ok=True)
        model_path='outputs/model.pkl'
        joblib.dump(final_pipeline, model_path)
if __name__ == '__main__':
    main()


