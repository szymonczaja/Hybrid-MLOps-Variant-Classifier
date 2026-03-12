# ACMG Variant Pathogenicity Classifier

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Azure ML](https://img.shields.io/badge/Azure_ML-Cloud_Training-informational.svg)](https://azure.microsoft.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)](https://www.docker.com/)

A machine learning project designed to classify the pathogenicity of genetic variants (Pathogenic / Benign). It integrates medical domain knowledge (ACMG guidelines) with an XGBoost model and a containerized cloud architecture.

**LIVE DEMO (Swagger UI):** [https://acmg-variant-api.azurewebsites.net/docs](https://acmg-variant-api.azurewebsites.net/docs)
*(To test the API, send one of the provided variant payloads via the POST `/predict` endpoint).*

---

## About the Project & ACMG Rules

Genetic variant classification requires the incorporation of specific medical guidelines, such as the ACMG (American College of Medical Genetics and Genomics) rules. These guidelines assess factors like the population frequency of a mutation and its potential impact on protein structure.

In this project, domain knowledge is embedded directly into the data pipeline. Using custom Scikit-Learn transformers, input data is first mapped and filtered according to ACMG rules, and the engineered features are then passed to the XGBoost classifier.

## Hybrid MLOps Architecture

The pipeline is divided into two independent parts to separate the training phase from the production environment:

1. **Cloud Training (Azure ML Jobs):** Training scripts execute computations on dedicated Azure Machine Learning clusters. The **Optuna** library is used for advanced hyperparameter optimization. All experiments, metrics, and final models are logged and versioned using **MLflow**.
2. **Independent Serving (Docker + FastAPI):** Instead of relying on built-in ML platform endpoints, the model is containerized. The API, built with FastAPI and packaged in a Docker image, ensures portability and allows for seamless deployment on services like Azure App Service.

## Repository Structure & Research Phase

* **`notebooks/ETL_Data_Preparation_notebook_v1.ipynb`** – Script responsible for raw data acquisition and preparation. This includes downloading the ClinVar database dump, file parsing, and variant annotation.
* **`notebooks/acmg_eda_fe_analysis.ipynb`** – Analytical notebook. It covers deep Exploratory Data Analysis (EDA), Feature Engineering, and the creation of Custom Transformers that implement the ACMG logic. Model evaluation is also performed here.
* **`src/`** – Production source code. Contains the training script for the Azure environment (`train_model.py`), the final transformer classes for the Scikit-Learn pipeline (`fe_transformers.py`), and the REST API microservice code (`app.py`).

## Model Results & Stability Analysis

The optimized XGBoost model achieved a **ROC AUC of 0.99** on the test set.

However, during data exploration, it was observed that one medical feature strongly dominated the predictions. To ensure the model had learned complex rules rather than just applying a simple threshold to this single variable, an additional robustness check was performed.

Calibration Curves were utilized to verify the model's behavior both with this feature included and after its removal. The results confirmed that the model did not overfit to a single parameter and that the returned probabilities are well-calibrated.

<img width="823" height="608" alt="Zrzut ekranu 2026-03-12 221015" src="https://github.com/user-attachments/assets/d4819daf-f436-4a61-8a3b-5754125859f7" />

<img width="801" height="597" alt="Zrzut ekranu 2026-03-12 221028" src="https://github.com/user-attachments/assets/f52c1084-0324-43d5-8c1f-ef5d1148e549" />


## Solved Engineering Challenges

During the containerization and deployment of the model to the production server, several typical environment-related challenges were addressed:

* **Data Format Consistency:** Standard Scikit-Learn pipelines cast Pandas DataFrames to Numpy arrays, which strips the column names required by XGBoost. This was resolved by applying a global configuration: `set_config(transform_output="pandas")`.
* **Deserialization & Namespace:** The serialized model (`model.pkl`) required access to the exact module paths from the training phase (`src` folder). In the server environment, memory-level module mapping (`sys.modules`) was implemented to prevent import errors without the need to retrain the model.
* **Dependency Management (Environment Drift):** Strict version pinning of libraries (e.g., `scikit-learn==1.8.0`) in the `requirements.txt` file and the base Docker image eliminated backend initialization errors associated with older versions of XGBoost.

## Local Deployment (Docker)

The project can be easily tested locally thanks to its fully containerized setup:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
   cd YOUR_REPOSITORY
   ```

2. **Build the Docker image:**
   ```bash
   docker build -t acmg-api .
   ```

3. **Run the container:**
   ```bash
   docker run -p 8000:80 acmg-api
   ```

The API will be available at: `http://localhost:8000/docs`.

## Example Payloads for Testing

To verify the model's functionality, navigate to the Swagger UI (`/predict` endpoint) and paste one of the following full JSON datasets.

**Variant 1: Pathogenic**
A rare missense mutation in the BRCA1 gene with a high pathogenicity score.
```json
{
  "CHROM": "17",
  "POS": "41276045",
  "REF": "A",
  "ALT": "G",
  "GENE_SYMBOL": "BRCA1",
  "MC": "SO:0001583|missense_variant",
  "GENEINFO": "BRCA1:672",
  "CLNVC": "single_nucleotide_variant",
  "ORIGIN": "1",
  "CLNREVSTAT": "criteria_provided,_multiple_submitters,_no_conflicts",
  "CLNDN": "Hereditary_breast_and_ovarian_cancer_syndrome",
  "AF_EXAC": 0.00002,
  "AF_TGP": 0.00001,
  "AF_ESP": 0.00005,
  "gnomad_exome_af_af": 0.00002,
  "dbnsfp_phylop_100way_vertebrate_score": 8.1,
  "dbnsfp_revel_score": 0.94,
  "dbnsfp_interpro_domain": "Znf_RING_lis"
}
```

**Variant 2: Benign**
A very common synonymous mutation in the NRAS gene with a high population frequency.
```json
{
  "CHROM": "1",
  "POS": "115256529",
  "REF": "G",
  "ALT": "A",
  "GENE_SYMBOL": "NRAS",
  "MC": "SO:0001819|synonymous_variant",
  "GENEINFO": "NRAS:4893",
  "CLNVC": "single_nucleotide_variant",
  "ORIGIN": "1",
  "CLNREVSTAT": "reviewed_by_expert_panel",
  "CLNDN": "not_specified",
  "AF_EXAC": 0.45,
  "AF_TGP": 0.42,
  "AF_ESP": 0.48,
  "gnomad_exome_af_af": 0.46,
  "dbnsfp_phylop_100way_vertebrate_score": -0.2,
  "dbnsfp_revel_score": 0.03,
  "dbnsfp_interpro_domain": "Ras_family"
}
```

<img width="1239" height="873" alt="Zrzut ekranu 2026-03-12 221055" src="https://github.com/user-attachments/assets/9813998a-8223-482b-a90b-e9d97ccdccc8" />
<img width="1152" height="666" alt="Zrzut ekranu 2026-03-12 221106" src="https://github.com/user-attachments/assets/5c07090f-f616-46b7-9701-74115fe88575" />
<img width="1124" height="426" alt="Zrzut ekranu 2026-03-12 221124" src="https://github.com/user-attachments/assets/c7f14a61-5f5d-4be5-8b46-2999466fea5a" />

   
