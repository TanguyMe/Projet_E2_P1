import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import shap

st.write("""
# Prédiction du prix de vente des biens immobiliers à Ames (Iowa USA)
""")
st.write('---')


X = pd.read_csv("clean_X.csv")


# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Quels sont vos critères?')


def user_input_features():
    Age = st.sidebar.slider('Ancienneté du bien', int(X.Age.min()), int(X.Age.max()), int(X.Age.mean()))
    GrLivArea = st.sidebar.slider('Surface au sol', int(X.GrLivArea.min()), int(X.GrLivArea.max()), int(X.GrLivArea.mean()))
    FullBath = st.sidebar.slider('Nombre de salles de bains', int(X.FullBath.min()), int(X.FullBath.max()), int(X.FullBath.mean()))
    SumSF = st.sidebar.slider('Indicateur de surface', int(X.SumSF.min()), int(X.SumSF.max()), int(X.SumSF.mean()), help='Somme de TotalBsmtSF et 1stFlrSF')
    GarageCars = st.sidebar.slider('Nombre de places du garage', int(X.GarageCars.min()), int(X.GarageCars.max()), int(X.GarageCars.mean()))
    MultQual = st.sidebar.slider('Indicateur de qualité', int(X.MultQual.min()), int(X.MultQual.max()), int(X.MultQual.mean()), help='Produit deOverallQual, ExterQual et KitchenQual')

    data = {'Age': Age,
            'GrLivArea': GrLivArea,
            'FullBath': FullBath,
            'SumSF': SumSF,
            'GarageCars': GarageCars,
            'MultQual': MultQual
            }
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Précisez vos critères')
st.write(df)
st.write('---')

loaded_model = pickle.load(open("finalized_model.sav", 'rb'))


# Apply Model to Make Prediction
def predict(data):
    prediction = loaded_model.predict(data)
    return '${:,}'.format(int(prediction))


st.header('Prediction du prix de vente')
st.write(predict(df))
st.write('---')

st.header('Impact des critères sur le prix de vente: Comparaison à la moyenne')
def get_shap_values(data):
    # set the tree explainer as the model of the pipeline
    explainer = shap.TreeExplainer(loaded_model['model'])
    # apply the preprocessing to x_test
    observations = loaded_model['preprocess'].transform(data)
    # get Shap values from preprocessed data
    return explainer.shap_values(observations)

def display_value(key, value):
    value = round(value)
    if value >= 0:
        return st.success(f'Le critère {key} augmente la valeur de ${value}')
    return st.error(f'Le critère {key} diminue la valeur de ${value}')

shap_values = get_shap_values(df)
cols = ['MultQual', 'GrLivArea', 'GarageCars', 'SumSF', 'FullBath', 'Age']
for i in range(len(shap_values[0])):
    display_value(cols[i], shap_values[0][i])

