# Vision-Language Model for Differential Diagnosis of Alzheimer's Disease Progression based on Multi-Modal Data Integration

This repository contains the official implementation for the paper:
**_"Vision-Language Model for Differential Diagnosis of Alzheimer's Disease Progression based on Multi-Modal Data Integration"_**

The code demonstrates the integration of MRI, PET, and clinical data using a Vision-Language Model (VLM) with the Unified Multi-Modal Attention (UMMA) mechanism and It employs abnormality tokens to enhance feature extraction from MRI and PET images for the diagnosis of Alzheimer's disease.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training the Model](#training-the-model)
- [Inference using Jupyter Notebook](#inference-using-jupyter-notebook)

## Overview

This project aims to develop a Vision-Language Model for integrating multi-modal data to enhance the differential diagnosis of Alzheimer's Disease Progression. The model uses:

1. MRI, PET, demographic information, and neuropsychological scales data as inputs.
2. Learnable abnormality tokens to capture structural and metabolic irregularities.
3. Unified Multi-Modal Attention (UMMA) to capture interactions the different modalities.

## Requirements

To run the code, you will need the following dependencies:

- Python 3.8+
- PyTorch==2.0.1
- transformers==4.37.2

You can install all necessary dependencies with:

```bash
pip install -r requirements.txt
```

## Installation

Clone this repository to your local machine:

```python
git clone https://github.com/your-username/alzheimers-diagnosis-vlm.git
cd alzheimers-diagnosis-vlm
```

## Data Preparation

Each data sample will be represented in a JSON format, where you link the MRI image, PET image, and corresponding clinical data. Below is an example of the JSON structure:

```json
{
    "patient_id": "001",
    "image": ["MRI.png", "PET.png"],
    "mri_label": 1,
    "pet_label": 1, 
    "conversations": [{"from": "USER", 
                      "value": "<image>\n<image>\nThe age of the patient is 86.4, the gender is Female. The Mini-Mental State Examination score is 20 out of 30, and the Clinical Dementia Rate score is 2.0. What the Alzheimer's Disease diagnosis result is? Select from the following: Cognitive Normal (CN), Mild Cognitive Impairment (MCI), Alzheimer's Disease (AD)."}, 
                      {"from": "ASSISTANT", 
                      "value": "Given the demographic information, neuropsychological assessment scores, and the image information, the diagnosis is Alzheimer's Disease."}]
}
```

## Training the Model

You can adjust training parameters in the `train.sh` file. It includes options for config for deepseed.

## Inference using Jupyter Notebook

After training, use the following script to evaluate the model on the test set:

1. **Open the notebook**:
   Navigate to the `inference.ipynb` file in the repository and open it in Jupyter Notebook or Jupyter Lab.
2. **Load your data** :
   Make sure your data (MRI, PET images, and clinical information) is prepared and organized as described in the [Dataset Preparation](#dataset-preparation) section.
3. **Modify paths and parameters** :
   Inside the notebook, ensure that the file paths for your model checkpoint are correctly set. If needed, modify the parameters such as the path to your dataset, model weights, and any configuration settings.\
4. **Run the notebook** :
   Execute the cells step-by-step to load the pre-trained model, preprocess the input data, and run inference. The results, including the model's predictions, will be displayed within the notebook.
