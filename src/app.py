import pandas as pd
import joblib
import sys
import os 
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.fe_transformers import (
        GeneRiskEstimator, AddAcmgRules, OriginRareLabelEncoder, 
        ZeroImputer, VariantsAtTributeExtractor, ImpactScoreEncoder
    )
except ImportError:
    from fe_transformers import (
        GeneRiskEstimator, AddAcmgRules, OriginRareLabelEncoder, 
        ZeroImputer, VariantsAtTributeExtractor, ImpactScoreEncoder
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("\n" + "="*50)
    print("🚀 START: Inicjalizacja serwisu ACMG...")

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__)) 
        root_dir = os.path.dirname(current_dir) 
        
        model_path = os.path.join(root_dir, "model.pkl") 
        
        if not os.path.exists(model_path):
            model_path = os.path.join(current_dir, "model.pkl")

        print(f"Szukam modelu w: {model_path}")

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("Model załadowany POMYŚLNIE!")
            print("Swagger UI dostępny pod: http://127.0.0.1:8000/docs")
        else:
            print(f"BŁĄD: Nie znaleziono pliku {model_path}")
            
    except Exception as e:
        print(f"KRYTYCZNY BŁĄD ładowania modelu: {e}")

    print("="*50 + "\n")
    
    yield  
    
    print("\nZamykanie serwisu...")
    model = None

app = FastAPI(title='ACMG Variant Classifier', version='1.0', lifespan=lifespan)

class VariantInput(BaseModel):
    CHROM: str
    POS: Union[str, int] 
    REF: str
    ALT: str
    GENE_SYMBOL: str
    
    MC: Optional[str] = None
    GENEINFO: Optional[str] = None
    CLNVC: Optional[str] = "Unknown"  
    ORIGIN: Optional[Union[str, int]] = "1" 
    CLNREVSTAT: Optional[str] = "no_assertion" 
    CLNDN: Optional[str] = "not_specified" 
    
    AF_EXAC: Optional[Union[float, str]] = None
    AF_TGP: Optional[Union[float, str]] = None
    AF_ESP: Optional[Union[float, str]] = None
    
    gnomad_exome_af_af: Optional[float] = None
    dbnsfp_phylop_100way_vertebrate_score: Optional[float] = None
    dbnsfp_revel_score: Optional[float] = None
    dbnsfp_interpro_domain: Optional[str] = None

    class Config:
        extra = "ignore"

@app.post("/predict")
def predict(data: VariantInput):
    if not model:
        raise HTTPException(status_code=503, detail="Model nie jest załadowany")
    
    try:
        input_dict = data.dict()
        column_mapping = {
            'gnomad_exome_af_af': 'gnomad_exome.af.af',
            'dbnsfp_phylop_100way_vertebrate_score': 'dbnsfp.phylop.100way_vertebrate.score',
            'dbnsfp_revel_score': 'dbnsfp.revel.score',
            'dbnsfp_interpro_domain': 'dbnsfp.interpro.domain'
        }
        
        for py_name, df_name in column_mapping.items():
            if py_name in input_dict:
                val = input_dict.pop(py_name) 
                input_dict[df_name] = val     

        
        df = pd.DataFrame([input_dict])
        df['POS'] = df['POS'].astype(str) 
        df['CHROM'] = df['CHROM'].astype(str)
        if 'ORIGIN' in df.columns:
             df['ORIGIN'] = df['ORIGIN'].astype(str) 

        prediction = model.predict(df)[0]
        try:
            proba = model.predict_proba(df)[0][1]
        except:
            proba = None
            
        result = "Pathogenic" if prediction == 1 else "Benign"
        
        return {
            "prediction": result,
            "probability": float(proba) if proba is not None else "N/A",
            "details": {
                "gene": data.GENE_SYMBOL,
                "variant": f"{data.CHROM}:{data.POS} {data.REF}>{data.ALT}"
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Prediction Error: {str(e)}")