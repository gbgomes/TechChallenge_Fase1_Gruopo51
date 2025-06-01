import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder, RobustScaler, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score, KFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score, mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from melhores_modelos import MelhoresModelos
import joblib
import os

# lendo o dataset
dados = pd.read_csv('fase_1/Tech_Challenge/insurance.csv', sep=',')

## Ajustando as colunas categóricas
# As colunas sex e smoker são binária, por isso usaremos o LabelEncoder
label_encoder_sex = LabelEncoder()
label_encoder_smoker = LabelEncoder()
label_encoder_region = LabelEncoder()
dados['sex'] = label_encoder_sex.fit_transform(dados['sex'])
dados['smoker'] = label_encoder_smoker.fit_transform(dados['smoker'])

# A coluna region possui 4 categorais. Assim, usaremos o OneHotEncoder,
# separando cada categoria em 1 nova coluna
ohe = OneHotEncoder()
regiao = ohe.fit_transform(dados[['region']])
regiao = regiao.toarray()
regiao_labels = ohe.categories_
regiao = pd.DataFrame(regiao, columns=regiao_labels[0])
dados = pd.concat([dados, regiao], axis=1)
dados.drop('region', axis=1, inplace=True)

dados_sem_outliers = dados
## Preenchendo dataframe sem outliers para ser usado nos modelos mais sensíveis a este tipo de dado
## Vamos fazer isso para as colunas bmi e charges conforme análise realizada no notebook
threshold = 3
dados['outlier'] = 0  # Inicializa a coluna 'outlier' com 0
z = np.abs((dados['bmi'] - dados['bmi'].mean()) / dados['bmi'].std())
dados.loc[z >= threshold, 'outlier'] = 1  # Marca outliers com 1
z = np.abs((dados['charges'] - dados['charges'].mean()) / dados['charges'].std())
dados.loc[z >= threshold, 'outlier'] = 1  # Marca outliers com 1
# Criação do dataframe sem os outliers e exclusão da coluna outliers dos dataframes
dados_sem_outliers = dados[dados['outlier'] == 0]
dados_sem_outliers.drop('outlier', axis=1, inplace=True)
dados.drop('outlier', axis=1, inplace=True)

## Separando as features do label
feature_labels = ['age', 'sex', 'bmi', 'children', 'smoker'] # Não inclui as colunas de região pois não são relevantes para o modelo
x = dados[feature_labels]
y = dados['charges']
x_sem_outlier = dados_sem_outliers[feature_labels]
y_sem_outlier = dados_sem_outliers['charges']

## Normalizando os dados com StandardScaler que trouxe melhores resultados que o MimMax scaler e o RobustScaler
scaler = StandardScaler()
x_nomalizado = scaler.fit_transform(x) 
x_sem_outlier_nomalizado = scaler.fit_transform(x_sem_outlier) 

## Separando 5% para os testes finais do modelo com dados nunca vistos pois usaremos o cross_val_score que faz separação interna dos dados de treino e teste
### como a maior correlação é com a coluna smoker, vamos estratificar por esta coluna
x_train, x_test, y_train, y_test = train_test_split(x_nomalizado, y, test_size=0.05, stratify=dados['smoker'], random_state=42)
x_train_sem_outlier, x_test_sem_outlier, y_train_sem_outlier, y_test_sem_outlier = train_test_split(x_sem_outlier_nomalizado, y_sem_outlier, test_size=0.05, random_state=42)

# Configuração de KFold.
kfold  = KFold(n_splits=10, shuffle=True, random_state=42)

# Instanciando a classe MelhoresModelos
mm = MelhoresModelos(kfold)

coletar_melhores_modelos = True
#coletar_melhores_modelos = False

if coletar_melhores_modelos:
    # Regressão linear
    lr = LinearRegression()
    lr.fit(x_train_sem_outlier, y_train_sem_outlier)

    # Verificando os melhores parâmetros para o KNR
    knr = mm.otimizar_knr(x_train_sem_outlier, y_train_sem_outlier)

    # Verificando os melhores parâmetros para o SVR
    svr = mm.otimizar_svr(x_train_sem_outlier, y_train_sem_outlier)

    # Verificando os melhores parâmetros para o GB
    gb = mm.otimizar_gradient_boosting(x_train, y_train)

    # Verificando os melhores parâmetros para o ridge
    rd = mm.otimizar_ridge(x_train_sem_outlier, y_train_sem_outlier)

    # Verificando os melhores parâmetros para o Decision Tree
    dt = mm.otimizar_decision_tree(x_train, y_train)
    
    # Verificando os melhores parâmetros para o RF
    rf = mm.otimizar_random_forest(x_train, y_train)
    
    # Verificando os melhores parâmetros para a Regressão Polinomial
    pp = mm.otimizar_poly_pipeline(x_train_sem_outlier, y_train_sem_outlier)

else:   
    knr_params  = {'metric': 'euclidean', 'n_neighbors': 11, 'weights': 'distance', 'p': 1} 
    svr_params  = {'C': 100, 'epsilon': 0.1, 'gamma': 'scale', 'kernel': 'linear'}
    gb_params   = {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50, 'random_state': 42, 'subsample': 1.0}
    rd_params   = {'alpha': 10.0}
    dt_params   = {'criterion': 'squared_error', 'max_depth': 5, 'max_features': None, 'min_samples_leaf': 10, 'min_samples_split': 2}
    rf_params   = {'max_depth': 10, 'max_features': None, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 200}
    poly_params = {'poly_degree': 2}
        
    # K-Nearest Neighbors para Regressão (KNR):
    knr = KNeighborsRegressor(n_neighbors=knr_params['n_neighbors'], metric=knr_params['metric'], weights=knr_params['weights'])
    knr.fit(x_train_sem_outlier, y_train)

    # Support Vector Regression (SVR)
    svr = SVR(C=svr_params['C'], epsilon=svr_params['epsilon'], gamma=svr_params['gamma'], kernel=svr_params['kernel'])
    svr.fit(x_train_sem_outlier, y_train)

    # Modelos de Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=gb_params['n_estimators'], learning_rate=gb_params['learning_rate'], max_depth=gb_params['max_depth'], random_state=gb_params['random_state'], subsample=gb_params['subsample'])
    gb.fit(x_train, y_train)

    # Regressão Ridge
    rd = Ridge(alpha = rd_params['alpha'])
    rd.fit(x_train_sem_outlier, y_train)

    ## Modelos em Árvores
    # Árvore de decisão 
    dt = DecisionTreeRegressor(criterion=dt_params['criterion'], max_depth=dt_params['max_depth'], max_features=dt_params['max_features'], min_samples_leaf=dt_params['min_samples_leaf'], min_samples_split=dt_params['min_samples_split'], random_state=42)
    dt.fit(x_train, y_train)

    # Random Forest 
    rf = RandomForestRegressor(n_estimators=rf_params['n_estimators'], max_depth=rf_params['max_depth'], min_samples_split=rf_params['min_samples_split'], min_samples_leaf=rf_params['min_samples_leaf'], random_state=42) 
    rf.fit(x_train, y_train)

    # Regressão linear
    lr = LinearRegression()
    lr.fit(x_train_sem_outlier, y_train)

    ## Regressão polinomial
    # Criar o pipeline para a regressão polinomial
    pp = Pipeline([
        ('poly', PolynomialFeatures(degree=poly_params['poly_degree'])), # Ou o grau que você definiu
        ('linear_regression', LinearRegression())
    ])

# executando a validação cruzada
lr_result    = cross_val_score(lr,  x_train_sem_outlier, y_train_sem_outlier, cv = kfold, scoring='r2')
poly_result  = cross_val_score(pp,  x_train_sem_outlier, y_train_sem_outlier, cv = kfold, scoring='r2')
ridge_result = cross_val_score(rd,  x_train_sem_outlier, y_train_sem_outlier, cv = kfold, scoring='r2')
svr_result   = cross_val_score(svr, x_train_sem_outlier, y_train_sem_outlier, cv = kfold, scoring='r2')
knr_result   = cross_val_score(knr, x_train_sem_outlier, y_train_sem_outlier, cv = kfold, scoring='r2')
tree_result  = cross_val_score(dt,  x_train, y_train, cv = kfold, scoring='r2')
rf_result    = cross_val_score(rf,  x_train, y_train, cv = kfold, scoring='r2')
gb_result    = cross_val_score(gb,  x_train, y_train, cv = kfold, scoring='r2')

# Coletando os resultados e combinando com os modelos treinados
modelos = {
    "Linear Regression": {"modelo": lr, "score_cv": lr_result.mean()},
    "Polynomial Regression": {"modelo": pp, "score_cv": poly_result.mean()},
    "Ridge Regression": {"modelo": rd, "score_cv": ridge_result.mean()},
    "Decision Tree Regressor": {"modelo": dt, "score_cv": tree_result.mean()},
    "Random Forest Regressor": {"modelo": rf, "score_cv": rf_result.mean()},
    "Gradient Boosting Regressor": {"modelo": gb, "score_cv": gb_result.mean()},
    "Support Vector Regression": {"modelo": svr, "score_cv": svr_result.mean()},
    "K-Neighbors Regressor": {"modelo": knr, "score_cv": knr_result.mean()}
}

# Imprimindo os resultados da validação cruzada (CV) de todos os modelos
print("\n--- Desempenho dos Modelos (R² Médio CV) ---")
for nome, dados_modelo in modelos.items():
    print(f"{nome}: {dados_modelo['score_cv']:.4f}")

# Encontra o melhor modelo com base no score da validação cruzada.
# O melhor modelo é o com o resultado mais alto de R2
melhor_modelo_nome = max(modelos, key=lambda k: modelos[k]['score_cv'])
melhor_modelo_dados = modelos[melhor_modelo_nome]
print(f"\nO melhor modelo (baseado no CV) é: {melhor_modelo_nome} com R² médio: {melhor_modelo_dados['score_cv']:.4f}")

# Escolhido o melhor modelo, executamos nele os 5% separados para testes
print(f"\nTestando o melhor modelo ({melhor_modelo_nome}) com dados nunca vistos...")
# Pega o objeto do modelo treinado (ou o pipeline treinado, no caso do POLY)
modelo_final_obj = melhor_modelo_dados['modelo']
# Fazer previsões no conjunto de teste
y_predito_final = modelo_final_obj.predict(x_test)

# Calcular e imprimir métricas do conjunto de teste
r2_final = r2_score(y_test, y_predito_final)
mse_final = mean_squared_error(y_test, y_predito_final)
rmse_final = root_mean_squared_error(y_test, y_predito_final) 
mae_final = mean_absolute_error(y_test, y_predito_final)

print(f'\nDesempenho do modelo {melhor_modelo_nome} no Conjunto de Teste:')
print(f"R²: {r2_final:.4f}")
print(f"Mean Squared Error (MSE): {mse_final:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_final:.4f}")
print(f"Mean Absolute Error (MAE): {mae_final:.4f}") ## erro na mesma escala que a variável target
print()
print()

# Salvando o modelo
model_filename = os.path.join('./fase_1/Tech_Challenge', 'melhor_modelo.joblib')
joblib.dump(modelo_final_obj, model_filename)
print(f"Modelo salvo em: {model_filename}")

# Salvando o scaler
scaler_filename = os.path.join('./fase_1/Tech_Challenge', 'scaler.joblib')
joblib.dump(scaler, scaler_filename)
print(f"Scaler salvo em: {scaler_filename}")

# Salvando a ordem das features
feature_order_filename = os.path.join('./fase_1/Tech_Challenge', 'feature_order.joblib')
joblib.dump(feature_labels, feature_order_filename)
print(f"Ordem das features salva em: {feature_order_filename}")

# Salvando os encoders
label_encoder_sex_filename = os.path.join('./fase_1/Tech_Challenge', 'label_encoder_sex.joblib')
joblib.dump(label_encoder_sex, label_encoder_sex_filename)
label_encoder_smoker_filename = os.path.join('./fase_1/Tech_Challenge', 'label_encoder_smoker.joblib')
joblib.dump(label_encoder_smoker, label_encoder_smoker_filename)
ohe_region_filename = os.path.join('./fase_1/Tech_Challenge', 'ohe_region.joblib')
joblib.dump(ohe, ohe_region_filename)
