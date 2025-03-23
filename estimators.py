import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Optional, Tuple, Union, List

# --- PCA

def pca(cov): 

    '''
    IN: return covariance matrix
    OUT: betas, eigenvalues, explained variance per factor, cumulative explained variance

    This function doesn't discard components for you, you'll have to do that yourself. 
    '''

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort descending
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Explained variance  
    explained_var = eigenvalues / eigenvalues.sum()
    explained_var_cumulative = np.cumsum(explained_var)

    # Compute loadings
    loadings = eigenvectors * np.sqrt(eigenvalues)

    # Assemble into loadings matrix
    betas = pd.DataFrame(
        loadings,
        index=cov.index,  
        columns=[f'factor_{i}' for i in range(len(eigenvalues))]
    )

    return(betas, eigenvalues, explained_var, explained_var_cumulative)

# --- MLE

def joreskog(cov, n_factors=None, max_iter=1000000, tol=1e-6, min_communal=1e-6):
    """
    Robust implementation of Jöreskog's factor analysis algorithm 
    with an explained variance measure relative to the original data.

    Parameters:
        cov (np.ndarray or pd.DataFrame): Symmetric covariance (or correlation) matrix.
        n_factors (int, optional): Desired number of factors; if None, estimated via eigenvalue threshold.
        max_iter (int): Maximum iterations to converge.
        tol (float): Convergence tolerance (relative changes in beta and psi).
        min_communal (float): Lower bound for psi (uniqueness) to guarantee positivity.
        
    Returns:
        betas (pd.DataFrame): Factor loadings with rows corresponding to variables
                              and exactly n_factors columns.
        factor_variances (np.ndarray): Sum of squared loadings per factor.
        explained_var (np.ndarray): Each factor's explained variance as a ratio of total original variance.
        explained_cumulative (np.ndarray): Cumulative explained variance.
    """
    # Convert DataFrame to numpy array immediately.
    original_index = None
    if isinstance(cov, pd.DataFrame):
        original_index = cov.index
        cov = cov.to_numpy()
    else:
        original_index = [f'var_{i+1}' for i in range(cov.shape[0])]
    
    # Validate covariance matrix.
    if not isinstance(cov, np.ndarray):
        raise ValueError("Covariance matrix must be a numpy array or pandas DataFrame")
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance matrix must be square")
    if not np.allclose(cov, cov.T, atol=1e-9):
        raise ValueError("Covariance matrix must be symmetric")
    
    n_vars = cov.shape[0]
    
    # Use eigenvalue threshold to determine number of factors if not specified.
    eigvals_full, _ = np.linalg.eigh(cov)
    tol_eigen = 1e-9 * np.max(eigvals_full)
    if n_factors is None:
        n_factors = int(np.sum(eigvals_full > tol_eigen))
    else:
        if not isinstance(n_factors, int):
            raise ValueError("Number of factors must be an integer")
        if n_factors <= 0:
            raise ValueError("Number of factors must be positive")
        n_factors = min(n_factors, n_vars)
    
    # Perform eigen-decomposition using np.linalg.eigh.
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx][:n_factors]
    eigenvecs = eigenvecs[:, idx][:, :n_factors]
    
    # Initialize beta as the product of eigenvectors and sqrt of eigenvalues.
    beta = np.array(eigenvecs @ np.diag(np.sqrt(np.maximum(eigenvals, 0))))
    communalities = np.sum(beta**2, axis=1)
    psi = np.diag(np.maximum(np.diag(cov) - communalities, min_communal))
    
    iter_num = 0
    beta_change = np.inf
    psi_change = np.inf
    
    while iter_num < max_iter and (beta_change > tol or psi_change > tol):
        # Compute the model-implied covariance matrix.
        sigma = beta @ beta.T + psi
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError("Current sigma is singular; try increasing min_communal or check cov.")
    
        # Update beta using the iterative routine.
        middle = np.linalg.inv(np.eye(n_factors) + np.asarray(beta).T @ sigma_inv @ np.asarray(beta))
        beta_new = np.asarray(cov) @ sigma_inv @ np.asarray(beta) @ middle
        
        # Update psi (uniquenesses), ensuring non-negativity.
        communalities_new = np.sum(beta_new**2, axis=1)
        psi_new_diag = np.maximum(np.diag(cov) - communalities_new, min_communal)
        psi_new = np.diag(psi_new_diag)
    
        # Convergence check.
        beta_change = np.linalg.norm(beta_new - beta) / (np.linalg.norm(beta) + np.finfo(float).eps)
        psi_change = np.linalg.norm(np.diag(psi_new) - np.diag(psi)) / (np.linalg.norm(np.diag(psi)) + np.finfo(float).eps)
    
        beta = beta_new
        psi = psi_new
        iter_num += 1

    if iter_num == max_iter and (beta_change > tol or psi_change > tol):
        print("Warning: The algorithm did not converge within the maximum number of iterations.")
    
    # Compute factor variances, where each is the sum of squared loadings in that factor.
    factor_variances = np.sum(beta**2, axis=0)
    # Use the original data's total variance.
    total_data_variance = np.sum(np.diag(cov))
    explained_var = factor_variances / total_data_variance
    explained_cumulative = np.cumsum(explained_var)
    
    factor_columns = [f'factor_{i}' for i in range(n_factors)]
    betas_df = pd.DataFrame(beta, index=original_index, columns=factor_columns)
    
    return betas_df, factor_variances, explained_var, explained_cumulative

# --- APCA

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

class APCA:
    def __init__(self, n_factors: int):
        self.n_factors = n_factors
        self.factors_: np.ndarray = None
        self.loadings_: pd.DataFrame = None
        self._returns: np.ndarray = None
        self.r2_scores_: np.ndarray = None

    def fit(self, 
            returns: np.ndarray,
            tickers: list = None,
            max_iter: int = 10,
            tol: float = 1e-6) -> 'APCA':
        n, T = returns.shape
        if tickers is None:
            tickers = [f'Asset_{i}' for i in range(n)]

        self._returns = returns
        X = self._standardize(returns)  # Standardize each asset's time series
        D_prev = np.var(X, axis=1, ddof=1)

        for _ in range(max_iter):
            R_std = X / np.sqrt(D_prev[:, np.newaxis])
            F_hat, betas, alphas, D_new = self._iteration_step(R_std, X)

            if np.max(np.abs(D_new - D_prev)) < tol:
                break

            D_prev = D_new.copy()

        self.factors_ = self._orthogonalize_factors(F_hat)  # Orthogonalize factors again
        self.loadings_ = pd.DataFrame(
            betas, 
            index=tickers,
            columns=[f'factor_{i+1}' for i in range(self.n_factors)]
        )

        # Calculate R² scores
        self._calculate_r2()
        return self

    def _calculate_r2(self):
        print("Starting _calculate_r2")
        print("Factors shape:", self.factors_.shape)
        print("Returns shape:", self._returns.shape)
        
        r2_scores = np.zeros(self.n_factors)
        for i in range(self.n_factors):
            factor = self.factors_[:, i:i+1]
            print(f"Factor {i} shape:", factor.shape)
            r2_sum = 0
            for asset in range(self._returns.shape[0]):
                asset_returns = self._returns[asset, :]
                print(f"Asset {asset} returns shape:", asset_returns.shape)
                reg = LinearRegression(fit_intercept=True)
                reg.fit(factor, asset_returns)
                r2_sum += reg.score(factor, asset_returns)
            r2_scores[i] = r2_sum / self._returns.shape[0]

    @staticmethod
    def _standardize(X: np.ndarray) -> np.ndarray:
        return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

    def _iteration_step(self, R_std: np.ndarray, X: np.ndarray):
        print("R_std shape:", R_std.shape)
        U, S, Vt = np.linalg.svd(R_std, full_matrices=False)
        idx = np.argsort(S)[::-1][:self.n_factors]

        F_hat = (S[idx, np.newaxis] * Vt[idx]).T
        print("F_hat shape after SVD:", F_hat.shape)
        F_hat = F_hat / np.std(F_hat, axis=0, keepdims=True)
        print("F_hat shape after standardization:", F_hat.shape)

        n = X.shape[0]
        betas = np.zeros((n, self.n_factors))
        alphas = np.zeros(n)
        D_new = np.zeros(n)

        for i in range(n):
            betas[i], alphas[i], D_new[i] = self._fit_asset(X[i, :], F_hat)

        return F_hat, betas, alphas, D_new

    @staticmethod
    def _fit_asset(y: np.ndarray, F: np.ndarray):
        reg = LinearRegression(fit_intercept=True)
        reg.fit(F, y)
        residuals = y - reg.predict(F)
        return reg.coef_, reg.intercept_, np.var(residuals, ddof=1)

    def _orthogonalize_factors(self, F: np.ndarray) -> np.ndarray:
        print("F shape before orthogonalization:", F.shape)  # Should be (T, k)
        U, S, Vt = np.linalg.svd(F, full_matrices=False)
        idx = np.argsort(S)[::-1][:self.n_factors]
        # Use U instead of Vt to maintain the time dimension
        F_orth = U[:, :self.n_factors]  # This keeps the (T, k) shape
        print("F shape after orthogonalization:", F_orth.shape)
        return F_orth / np.std(F_orth, axis=0, keepdims=True)

def fit_apca(corr_matrix: np.ndarray,
             dataset,  # returns
             n_factors: int = 5,
             tickers: list = None) -> tuple:
    """Fit APCA model from a correlation matrix.
    
    Args:
        corr_matrix: correlation matrix of shape (n, n)
        T: number of time periods
        n_factors: number of factors to extract
        tickers: list of asset names
    """
    n = corr_matrix.shape[0]
    T = dataset.shape[0]

    eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = np.maximum(eigenvals[idx], 1e-10)
    eigenvecs = eigenvecs[:, idx]

    corr_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    # Generate T samples using Cholesky
    rng = np.random.default_rng()
    Z = rng.standard_normal((n, T))
    R = (np.linalg.cholesky(corr_matrix) @ Z) * np.sqrt(T)

    model = APCA(n_factors=n_factors)
    model.fit(R, tickers=tickers)

    return model.loadings_, model.factors_, eigenvals[:n_factors], model.r2_scores_

