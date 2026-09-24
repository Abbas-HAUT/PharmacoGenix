import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import joblib

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem

# Scikit-Learn imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
INPUT_PATH = r".csv"
OUTPUT_DIR = r"results"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. DATA LOADING & PREPROCESSING (RDKit)
# ==========================================
print("Loading data...")
df = pd.read_csv(INPUT_PATH)

print("Cleaning dataset and calculating Lipinski rules using RDKit...")
def process_smiles(smi):
    """Canonicalize SMILES and calculate Lipinski properties."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return pd.Series([None, None, None, None, None, None])
        
        canon_smiles = Chem.MolToSmiles(mol)
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        rot_bonds = Lipinski.NumRotatableBonds(mol)
        
        return pd.Series([canon_smiles, mw, logp, hbd, hba, rot_bonds])
    except:
        return pd.Series([None, None, None, None, None, None])

# Apply RDKit function
df[['canonical_smiles_clean', 'MW_calc', 'LogP_calc', 'HBD_calc', 'HBA_calc', 'RotBonds_calc']] = df['canonical_smiles'].apply(process_smiles)

# Drop invalid SMILES and Duplicates
df = df.dropna(subset=['canonical_smiles_clean'])
df = df.drop_duplicates(subset=['canonical_smiles_clean'])

# Remove molecules with MW > 1000
df = df[df['MW_calc'] <= 1000]

# ==========================================
# 3. IC50 NORMALIZATION & BIOACTIVITY CLASS
# ==========================================
print("Normalizing IC50 and classifying bioactivity...")

# Cap extreme outliers at 10^8 nM
df['standard_value_norm'] = df['standard_value'].clip(upper=1e8)

# Convert to pIC50: pIC50 = -log10(IC50 * 10^-9) = 9 - log10(IC50)
df['pIC50_values'] = 9 - np.log10(df['standard_value_norm'])

# Bioactivity Classification
def classify_activity(ic50):
    if ic50 < 1000:
        return 'Active'
    elif ic50 >= 10000:
        return 'Inactive'
    else:
        return 'Intermediate'

df['Activity Class'] = df['standard_value_norm'].apply(classify_activity)

# Filter out intermediate compounds
df = df[df['Activity Class'] != 'Intermediate']

# Save preprocessed dataset
preprocessed_path = os.path.join(OUTPUT_DIR, 'preprocessed_dataset.csv')
df.to_csv(preprocessed_path, index=False)
print(f"Preprocessed data saved to: {preprocessed_path}")

# ==========================================
# 4. HYPOTHESIS TESTING (Mann-Whitney U) & EDA
# ==========================================
print("Performing Mann-Whitney U tests & generating plots...")
features_to_test = ['MW_calc', 'LogP_calc', 'HBD_calc', 'HBA_calc', 'standard_value_norm', 'pIC50_values']

active_df = df[df['Activity Class'] == 'Active']
inactive_df = df[df['Activity Class'] == 'Inactive']

test_results = []

sns.set_theme(style="whitegrid")
for feature in features_to_test:
    # Statistical Test
    stat, p_val = mannwhitneyu(active_df[feature], inactive_df[feature], alternative='two-sided')
    test_results.append({'Feature': feature, 'U-Statistic': stat, 'p-value': p_val})
    
    # Visualization
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Activity Class', y=feature, data=df, palette='Set2')
    plt.title(f'Distribution of {feature} (p-value: {p_val:.2e})')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{feature}_boxplot.png'), dpi=300)
    plt.close()

# Save test results
pd.DataFrame(test_results).to_csv(os.path.join(OUTPUT_DIR, 'mann_whitney_results.csv'), index=False)

# ==========================================
# 5. FEATURE ENGINEERING (FINGERPRINTS) & ML
# ==========================================
print("Generating Binary Fingerprint Descriptors...")
def get_fingerprint(smi):
    mol = Chem.MolFromSmiles(smi)
    # Using Morgan Fingerprints as binary 0/1 descriptors
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return np.array(fp)

X = np.array([get_fingerprint(smi) for smi in df['canonical_smiles_clean']])
y = df['pIC50_values'].values

# Outlier Removal using IQR on Target Variable (pIC50)
Q1 = np.percentile(y, 25)
Q3 = np.percentile(y, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

valid_idx = (y >= lower_bound) & (y <= upper_bound)
X = X[valid_idx]
y = y[valid_idx]

# Missing Value Imputation (If any)
imputer = SimpleImputer(strategy='most_frequent') # Best for binary data
X = imputer.fit_transform(X)

# Variance Thresholding (threshold = 0.01)
print("Applying Variance Thresholding...")
selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)

# Save feature selector for Streamlit
joblib.dump(selector, os.path.join(OUTPUT_DIR, 'variance_selector.pkl'))

# Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# ==========================================
# 6. RANDOM FOREST MODEL BUILDING & TUNING
# ==========================================
print("Training and Optimizing Random Forest Regressor...")
rf = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Evaluation
y_pred = best_rf.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Model Performance -> R2 Score: {r2:.4f} | RMSE: {rmse:.4f}")

# Save the trained model
model_path = os.path.join(OUTPUT_DIR, 'PharmacoGenix_RF_Model.pkl')
joblib.dump(best_rf, model_path)
print(f"Model saved to: {model_path}")

# ==========================================
# 7. GENERATE STREAMLIT APP
# ==========================================
print("Generating PharmacoGenix Streamlit App...")

streamlit_code = f"""import streamlit as st
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski

# Page Config
st.set_page_config(page_title="PharmacoGenix Predictor", layout="centered")

# Load Models
@st.cache_resource
def load_models():
    model = joblib.load(r"{model_path}")
    selector = joblib.load(r"{os.path.join(OUTPUT_DIR, 'variance_selector.pkl')}")
    return model, selector

rf_model, variance_selector = load_models()

def get_fingerprint_and_properties(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    
    # Fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp_array = np.array(fp).reshape(1, -1)
    
    # Properties
    props = {{
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'HBA': Lipinski.NumHAcceptors(mol),
        'Rotatable Bonds': Lipinski.NumRotatableBonds(mol)
    }}
    return fp_array, props

st.title("🧬 PharmacoGenix")
st.subheader("Predicting MAB Therapeutics pIC50 via Machine Learning")

st.markdown("Enter a **Canonical SMILES** string below to evaluate its molecular properties and predict its bioactivity (pIC50) using the optimized Random Forest Regressor.")

smiles_input = st.text_input("Enter SMILES string:", "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5")

if st.button("Predict pIC50"):
    if smiles_input:
        fp_array, properties = get_fingerprint_and_properties(smiles_input)
        
        if fp_array is not None:
            # Display properties
            st.write("### 🧪 Lipinski Properties Evaluated")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("MW", f"{{properties['MW']:.2f}}")
            col2.metric("LogP", f"{{properties['LogP']:.2f}}")
            col3.metric("HBD", properties['HBD'])
            col4.metric("HBA", properties['HBA'])
            col5.metric("Rot Bonds", properties['Rotatable Bonds'])
            
            # Predict
            try:
                # Apply variance thresholding
                fp_selected = variance_selector.transform(fp_array)
                
                # Predict
                prediction = rf_model.predict(fp_selected)[0]
                
                st.write("### 🎯 Prediction Results")
                st.success(f"Predicted pIC50: **{{prediction:.4f}}**")
                
                # Reverse math for IC50 (nM)
                predicted_ic50 = 10**(9 - prediction)
                if predicted_ic50 < 1000:
                    st.info(f"Estimated IC50: {{predicted_ic50:.2f}} nM (Likely **Active**)")
                elif predicted_ic50 >= 10000:
                    st.error(f"Estimated IC50: {{predicted_ic50:.2f}} nM (Likely **Inactive**)")
                else:
                    st.warning(f"Estimated IC50: {{predicted_ic50:.2f}} nM (**Intermediate**)")
                    
            except Exception as e:
                st.error(f"Error during prediction: {{e}}")
        else:
            st.error("Invalid SMILES string. Please check your input and try again.")
    else:
        st.warning("Please enter a SMILES string first.")
"""

app_path = os.path.join(OUTPUT_DIR, 'PharmacoGenix_app.py')
with open(app_path, "w", encoding="utf-8") as f:
    f.write(streamlit_code)

print(f"\n✅ Pipeline Complete! Streamlit App generated at: {app_path}")
print("To run the application, open your terminal and type:")
print(f'streamlit run "{app_path}"')
