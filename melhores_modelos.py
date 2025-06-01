import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, accuracy_score # accuracy_score não é usado para regressão, mas está no KNN original
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

class MelhoresModelos:
    """
    Classe para encapsular a otimização de hiperparâmetros de modelos de regressão
    usando GridSearchCV.
    """
    def __init__(self, kfold):
        """
        Inicializa o otimizador com uma estratégia de validação cruzada.

        Args:
            cv_strategy: Objeto de validação cruzada (ex: KFold, StratifiedKFold).
        """
        
        if kfold is None:
            raise ValueError("É necessário fornecer uma estratégia de validação cruzada (KFold).")
        self.cv = kfold


    def otimizar_knr(self, x, y):
        """Otimiza KNeighborsRegressor."""
        print("\n--- Otimizando KNeighborsRegressor ---")
        param_grid = {'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10, 11],
                      'weights': ['uniform', 'distance'],
                      'metric': ['euclidean', 'manhattan', 'minkowski'],
                      'p': [1, 2]} # Relevante se metric='minkowski'

        # Para regressão, usamos 'r2' ou outra métrica de regressão
        grid = GridSearchCV(KNeighborsRegressor(),
                            param_grid=param_grid,
                            scoring='r2', # Métrica correta para regressão
                            cv=self.cv,
                            n_jobs=-1, # -1 usa todos os processadores
                            verbose=1)
        grid.fit(x, y)
        print("Melhores parâmetros para KNN:", grid.best_params_)
        print(f"Melhor R² CV para KNN: {grid.best_score_:.4f}")
        return grid.best_estimator_

    def otimizar_random_forest(self, x, y):
        """Otimiza RandomForestRegressor."""
        print("\n--- Otimizando RandomForestRegressor ---")
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        grid_search = GridSearchCV(RandomForestRegressor(random_state=42),
                                   param_grid=param_grid,
                                   scoring='r2',
                                   cv=self.cv,
                                   n_jobs=-1,
                                   verbose=1)
        grid_search.fit(x, y)
        print("Melhores parâmetros para RandomForest:", grid_search.best_params_)
        print(f"Melhor R² CV para RandomForest: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_

    def otimizar_gradient_boosting(self, x, y):
        """Otimiza GradientBoostingRegressor."""
        print("\n--- Otimizando GradientBoostingRegressor ---")
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
        grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42),
                                   param_grid=param_grid,
                                   scoring='r2',
                                   cv=self.cv,
                                   n_jobs=-1,
                                   verbose=1)
        grid_search.fit(x, y)
        print("Melhores parâmetros para Gradient Boosting:", grid_search.best_params_)
        print(f"Melhor R² CV para Gradient Boosting: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_

    def otimizar_ridge(self, x, y):
        """Otimiza Ridge Regression."""
        print("\n--- Otimizando Ridge Regression ---")
        param_grid = {
            'alpha': np.logspace(-3, 3, 7) # 7 valores de 0.001 a 1000
        }
        grid_search = GridSearchCV(Ridge(),
                                   param_grid=param_grid,
                                   scoring='r2',
                                   cv=self.cv,
                                   n_jobs=-1,
                                   verbose=1)
        grid_search.fit(x, y)
        print("Melhores parâmetros para Ridge:", grid_search.best_params_)
        print(f"Melhor R² CV para Ridge: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_

    def otimizar_svr(self, x, y):
        """Otimiza Support Vector Regression (SVR)."""
        print("\n--- Otimizando Support Vector Regression (SVR) ---")
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 1],
            'kernel': ['rbf', 'linear'],
            'epsilon': [0.1, 0.2]
        }
        grid_search = GridSearchCV(SVR(),
                                   param_grid=param_grid,
                                   scoring='r2',
                                   cv=self.cv,
                                   n_jobs=-1,
                                   verbose=1)
        grid_search.fit(x, y)
        print("Melhores parâmetros para SVR:", grid_search.best_params_)
        print(f"Melhor R² CV para SVR: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_

    def otimizar_decision_tree(self, x, y):
        """Otimiza DecisionTreeRegressor."""
        print("\n--- Otimizando Decision Tree Regressor ---")
        param_grid = {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 3, 5, 10],
            'max_features': [None, 'sqrt', 'log2']
        }
        grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42),
                                   param_grid=param_grid,
                                   scoring='r2',
                                   cv=self.cv,
                                   n_jobs=-1,
                                   verbose=1)
        grid_search.fit(x, y)
        print("Melhores parâmetros para Decision Tree:", grid_search.best_params_)
        print(f"Melhor R² CV para Decision Tree: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_

    def otimizar_poly_pipeline(self, x, y):
        """Otimiza Pipeline PolynomialFeatures + LinearRegression."""
        print("\n--- Otimizando Polynomial Regression Pipeline ---")
        poly_pipeline = Pipeline([
            ('poly', PolynomialFeatures()),
            ('linear_regression', LinearRegression())
        ])
        param_grid = {
            'poly__degree': [1, 2, 3],
            'linear_regression__fit_intercept': [True, False]
        }
        grid_search = GridSearchCV(poly_pipeline,
                                   param_grid=param_grid,
                                   scoring='r2',
                                   cv=self.cv,
                                   n_jobs=-1,
                                   verbose=1)
        grid_search.fit(x, y)
        print("Melhores parâmetros para Polynomial Pipeline:", grid_search.best_params_)
        print(f"Melhor R² CV para Polynomial Pipeline: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_

