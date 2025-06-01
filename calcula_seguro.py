import flask
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

app = flask.Flask(__name__)

# Carrega o modelo, encoders e etc
MODEL_DIR = 'modelo_deploy'
try:
    model = joblib.load('./fase_1/Tech_Challenge/melhor_modelo.joblib')
    scaler = joblib.load('./fase_1/Tech_Challenge/scaler.joblib')
    feature_order = joblib.load('./fase_1/Tech_Challenge/feature_order.joblib')
    label_encoder_sex = joblib.load('./fase_1/Tech_Challenge/label_encoder_sex.joblib')
    label_encoder_smoker = joblib.load('./fase_1/Tech_Challenge/label_encoder_smoker.joblib')
    ohe_region = joblib.load('./fase_1/Tech_Challenge/ohe_region.joblib')

except FileNotFoundError:
    print("Erro: Arquivos de modelo/scaler/features não encontrados.")
    model, scaler, feature_order = None, None, None


def preprocessamento(data):
    """Preprocessa os dados de entrada EXATAMENTE como no treinamento."""

    # Verifique todos os objetos carregados
    if not model or not scaler or not feature_order or not label_encoder_sex or not label_encoder_smoker or not ohe_region: 
         raise ValueError("Objetos de pré-processamento não carregados.")

    # Cria DataFrame com o dado recebido
    df = pd.DataFrame([data])

    ## Aplicar o mesmo pré-processamento do treino
    # Label Encoding
    df['sex']    = label_encoder_sex.transform(df['sex'])
    df['smoker'] = label_encoder_smoker.transform(df['smoker'])

    # One-Hot Encoding
    regiao = ohe_region.transform(df[['region']])
    regiao = regiao.toarray()
    regiao_labels = ohe_region.categories_
    regiao = pd.DataFrame(regiao, columns=regiao_labels[0])
    df = pd.concat([df, regiao], axis=1)
    df.drop('region', axis=1, inplace=True)

    # Garantir ordem e colunas faltantes
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0 # Adiciona colunas que podem faltar (ex: uma região específica)
    df = df[feature_order] # Reordena para bater com o treino

    # Scaling (usando o scaler carregado)
    scaled_data = scaler.transform(df)

    return scaled_data

@app.route('/calculaseguro', methods=['POST'])
def predict():
    """Endpoint para receber dados e retornar previsões."""

    if not model:
        return flask.jsonify({"error": "Modelo não disponível"}), 500

    try:
        input_data = flask.request.get_json(force=True)

        required_fields = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        if not all(field in input_data for field in required_fields):
             return flask.jsonify({"error": "Dados de entrada incompletos"}), 400

        # Pré-processar
        preproc = preprocessamento(input_data)

        # Prever
        resultado = model.predict(preproc)
        

        # Retornar
        return flask.jsonify({'predicted_charge': resultado[0]})

    except Exception as e:
        # Logar o erro é uma boa prática
        print(f"Erro na previsão: {e}")
        return flask.jsonify({"error": f"Erro interno: {e}"}), 500

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=False)
