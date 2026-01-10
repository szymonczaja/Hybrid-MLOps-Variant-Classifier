# Automated ACMG Variant Classification Engine 

## Live Demo
Check out the live application running on Render:
 **https://acmg-model.onrender.com/docs**


##  Business Context
Genetic variant interpretation is a significant bottleneck in precision medicine. The American College of Medical Genetics (ACMG) provides a strict, evidence-based framework for classifying variants (Pathogenic vs. Benign), but manual application of these rules is time-consuming, repetitive, and prone to human error.

##  Project Overview
This project implements an **automated classification pipeline** based on the official ACMG guidelines. It processes genetic variant data, applies specific evidence criteria (e.g., population frequency, computational predictions, functional studies), and determines the final pathogenicity status without manual intervention.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Containerization:** Docker
* **Deployment:** Render
* **Logic:** Rule-based algorithm implementation (Complex boolean logic for clinical guidelines)

## Key Features
* **Criteria Implementation:** Algorithmic mapping of key ACMG codes (e.g., PVS1 for null variants, PM2 for rare variants, BA1 for common variants).
* **Dynamic Scoring:** Automated calculation of the final classification (Pathogenic, Likely Pathogenic, VUS, Likely Benign, Benign) based on the combination of triggered rules.
* **Scalability:** Designed to handle batch processing of multiple variants simultaneously.

##  Impact
* **Efficiency:** Drastically reduces the time required for initial variant triage.
* **Standardization:** Eliminates inter-operator variability in applying subjective criteria.
* **Scalability:** Allows for high-throughput analysis of genomic datasets.

---

##  How to Run with Docker

If you want to run this application locally in a containerized environment, follow these steps:

### 1. Build the Docker Image
Navigate to the project directory in your terminal and run:
```bash
docker build -t acmg-classifier .

### 2. Run the Container

docker run -p 8000:8000 acmg-classifier

### 3. Access the App
Open your browser and go to: http://localhost:8000/docs
