# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 14:51:14 2025
Updated by AI on Mon Jun 2 15:00:00 2025
Further Updated by AI on Mon Jun 2 20:07:00 2025

@author: pc
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from urllib.parse import quote
import requests
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem
from tensorflow.keras.models import load_model
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from scipy.special import softmax
from PIL import Image
import io
import os
import tensorflow as tf

st.set_page_config(page_title="OEB Prediction Pro", layout="wide", page_icon="🔬")

try:
    DESC_NAMES = [desc[0] for desc in Descriptors._descList]
except AttributeError:
    st.warning("Could not dynamically load RDKit descriptor names. Using a predefined list might be necessary if errors occur.")
    DESC_NAMES = []

OEB_DESCRIPTIONS = {
    0: "No exposure limits: Minimal or no systemic toxicity.",
    1: "OEB 1: Low hazard (OEL: 1000 - 5000 µg/m³)",
    2: "OEB 2: Moderate hazard (OEL: 100 - 1000 µg/m³)",
    3: "OEB 3: High hazard (OEL: 10 - 100 µg/m³)",
    4: "OEB 4: Very high hazard (OEL: 1 - 10 µg/m³)",
    5: "OEB 5: Extremely high hazard (OEL: < 1 µg/m³)",
    6: "OEB 6: Extremely potent (OEL: < 0.1 µg/m³)"
}
MODEL_NAMES = ["MLP", "SVC", "XGBoost", "RandomForest", "DecisionTree"]
DEFAULT_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
MODEL_DIR = "models"

def get_model_path(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, MODEL_DIR, filename)

@st.cache_resource
def load_models_and_scalers():
    scalers = {}
    classifiers = {}
    cnn_model = None
    try:
        scalers = {
            "desc": joblib.load(get_model_path("scaler_descriptors.pkl")),
            "cnn_input": joblib.load(get_model_path("scaler_features_cnn.pkl")),
            "cnn_output": joblib.load(get_model_path("scaler_features_cnn_output.pkl"))
        }
        classifiers = {
            name: joblib.load(get_model_path(f"model_{name}.pkl")) for name in MODEL_NAMES
        }
        imported = tf.saved_model.load(get_model_path("cnn_model_tf213_compatible"))
        cnn_model = imported.signatures["serving_default"]
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}. Please ensure all model files are in the '{MODEL_DIR}' subdirectory.")
        return None, {}, {}
    except Exception as e:
        st.error(f"An unexpected error occurred while loading models: {e}")
        return None, {}, {}
    return cnn_model, scalers, classifiers

def compute_cnn_ready_features(smiles, scalers, cnn_model):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if not DESC_NAMES:
        st.error("Descriptor names (DESC_NAMES) are not defined. Cannot calculate descriptors.")
        return None

    desc_calc = MolecularDescriptorCalculator(DESC_NAMES)
    descriptors = np.array(desc_calc.CalcDescriptors(mol))
    padded_desc = np.zeros(1024)
    padded_desc[:min(len(descriptors), 1024)] = descriptors[:min(len(descriptors), 1024)]

    try:
        norm_desc = scalers["desc"].transform([padded_desc])[0]
    except Exception as e:
        st.error(f"Error scaling descriptors: {e}")
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp_as_numpy_array = np.zeros((1024,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, fp_as_numpy_array)
    combined_features = np.stack((norm_desc, fp_as_numpy_array), axis=-1)

    if combined_features.shape != (1024, 2):
        st.error(f"Unexpected shape for combined_features: {combined_features.shape}")
        return None

    try:
        cnn_input_image = combined_features.reshape(32, 32, 2)
        norm_input_flat = scalers["cnn_input"].transform(cnn_input_image.reshape(1, -1))
        norm_input_reshaped = norm_input_flat.reshape(1, 32, 32, 2)
    except Exception as e:
        st.error(f"Error processing CNN input: {e}")
        return None

    try:
        input_name = list(cnn_model.structured_input_signature[1].keys())[0]  # dynamically determine input key
        output_dict = cnn_model(**{input_name: tf.convert_to_tensor(norm_input_reshaped, dtype=tf.float32)})
        cnn_features_raw = list(output_dict.values())[0].numpy()
    except Exception as e:
        st.error(f"Error during CNN model inference: {e}")
        return None

    try:
        cnn_features_scaled = scalers["cnn_output"].transform(cnn_features_raw)
    except Exception as e:
        st.error(f"Error scaling CNN output: {e}")
        return None

    return cnn_features_scaled

def main():
    st.title("🔬 OEB Prediction Pro")
    cnn_model, scalers, classifiers = load_models_and_scalers()
    if cnn_model is None or not scalers or not classifiers:
        return

    smiles = st.text_input("Enter SMILES:", value=DEFAULT_SMILES)
    features = compute_cnn_ready_features(smiles, scalers, cnn_model)

    if features is not None:
        model = classifiers[MODEL_NAMES[0]]  # Use selected model
        try:
            probs = model.predict_proba(features)[0]
        except Exception as e:
            st.warning(f"predict_proba failed: {e}. Using fallback.")
            try:
                decision_scores = model.decision_function(features)
                if decision_scores.ndim == 1:
                    decision_scores = decision_scores.reshape(1, -1)
                probs = softmax(decision_scores, axis=1)[0]
            except Exception as e:
                probs = np.full(len(OEB_DESCRIPTIONS), 1 / len(OEB_DESCRIPTIONS))

        pred_class = int(np.argmax(probs))
        st.success(f"Predicted OEB: {pred_class} - {OEB_DESCRIPTIONS.get(pred_class, 'Unknown')}")
        df = pd.DataFrame({"Class": list(OEB_DESCRIPTIONS.keys()), "Probability": probs})
        st.bar_chart(df.set_index("Class"))

if __name__ == "__main__":
    main()
