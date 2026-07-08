import numpy as np
from sklearn.linear_model import LinearRegression


class SimpleGLMBoost:
    """
    Custom GLMBoost-inspired model for binary classification.
    """

    def __init__(self, n_estimators=1000, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        F = np.zeros(X.shape[0])

        for _ in range(self.n_estimators):
            prob = self._sigmoid(F)
            residuals = y - prob

            model = LinearRegression()
            model.fit(X, residuals)

            update = model.predict(X)
            F += self.learning_rate * update

            self.models.append(model)

        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        F = np.zeros(X.shape[0])

        for model in self.models:
            F += self.learning_rate * model.predict(X)

        prob = self._sigmoid(F)

        return np.vstack([1 - prob, prob]).T

    def predict(self, X, threshold=0.5):
        prob = self.predict_proba(X)[:, 1]
        return (prob >= threshold).astype(int)