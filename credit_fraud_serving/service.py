# Service
import pandas as pd
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.frameworks.xgboost import XgboostModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
import xgboost as xgb


@env(requirements_txt_file="./requirements.txt")
@artifacts([PickleArtifact('encoder'), XgboostModelArtifact('model')])
class FraudClassifier(BentoService):

    @api(input=DataframeInput(), batch=True)
    def predict(self, df: pd.DataFrame):
        columns = df.columns[1:-1].to_list()
        df[columns] = self.artifacts.encoder.transform(df[columns])
        return self.artifacts.model.predict(xgb.DMatrix(df))
