import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

class GaussianMixtureModelVectorized:
    """
    Vectorized Gaussian Mixture Model with numerical stability improvements.
    """
    
    def __init__(self, n_components=2, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self.mu_ = None
        self.cov_ = None
        self.pi_ = None
        self.log_likelihoods_ = None
        self.n_iter_ = None
        self.converged_ = False
    
    def fit(self, X, max_iter=500, tol=1e-6, initial_theta=None, verbose=True):
        self.max_iter = max_iter
        self.tol = tol
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        self._initialize_parameters(X, initial_theta)
        self.log_likelihoods_ = []
        
        for iteration in tqdm(range(self.max_iter), disable=not verbose):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)
            ll = self._compute_log_likelihood(X)
            self.log_likelihoods_.append(ll)
            
            if iteration > 0:
                if abs(self.log_likelihoods_[-1] - self.log_likelihoods_[-2]) < self.tol:
                    self.converged_ = True
                    self.n_iter_ = iteration + 1
                    break
        else:
            self.n_iter_ = self.max_iter
        
        return self
    
    def _set_parameters(self, mus, covs, pi):
        self.mu_ = np.array(mus)
        self.cov_ = np.array(covs)
        self.pi_ = np.array(pi)
    
    def _initialize_parameters(self, X, theta=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        if theta is not None:
            self._set_parameters(*theta)
        else:
            indices = np.random.choice(n_samples, self.n_components, replace=False)
            self._set_parameters(
                X[indices],
                np.array([np.eye(n_features) for _ in range(self.n_components)]),
                np.ones(self.n_components) / self.n_components
            )
    
    def _e_step(self, X):
        """Vectorized E-step with log-sum-exp trick for numerical stability."""
        n_samples = X.shape[0]
        log_responsibilities = np.zeros((n_samples, self.n_components))
        
        # Compute log(π_k * N(x_i | μ_k, Σ_k)) for all i,k
        for k in range(self.n_components):
            log_responsibilities[:, k] = (
                np.log(self.pi_[k]) + 
                multivariate_normal.logpdf(X, mean=self.mu_[k], cov=self.cov_[k])
            )
        
        # Log-sum-exp trick: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
        max_log = np.max(log_responsibilities, axis=1, keepdims=True)
        log_sum_exp = max_log + np.log(np.sum(np.exp(log_responsibilities - max_log), axis=1, keepdims=True))
        log_responsibilities -= log_sum_exp
        
        return np.exp(log_responsibilities)
    
    def _m_step(self, X, responsibilities):
        """Vectorized M-step."""
        n_samples, n_features = X.shape
        
        # N_k = sum of responsibilities for each component
        N_k = responsibilities.sum(axis=0)
        
        # Update mixing proportions
        self.pi_ = N_k / n_samples
        
        # Update means: μ_k = (1/N_k) * Σ_i γ_ik * x_i
        self.mu_ = (responsibilities.T @ X) / N_k[:, np.newaxis]
        
        # Update covariances
        self.cov_ = np.zeros((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = X - self.mu_[k]
            self.cov_[k] = (responsibilities[:, k:k+1] * diff).T @ diff / N_k[k]
            self.cov_[k] += 1e-6 * np.eye(n_features)
    
    def _compute_log_likelihood(self, X):
        """Compute log-likelihood using log-sum-exp trick."""
        n_samples = X.shape[0]
        log_likelihood_per_point = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            log_likelihood_per_point[:, k] = (
                np.log(self.pi_[k]) + 
                multivariate_normal.logpdf(X, mean=self.mu_[k], cov=self.cov_[k])
            )
        
        # Log-sum-exp trick
        max_log = np.max(log_likelihood_per_point, axis=1)
        log_likelihood = np.sum(
            max_log + np.log(np.sum(np.exp(log_likelihood_per_point - max_log[:, np.newaxis]), axis=1))
        )
        
        return log_likelihood
    
    def predict_proba(self, X):
        X = np.asarray(X)
        return self._e_step(X)
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
    def score(self, X):
        X = np.asarray(X)
        return self._compute_log_likelihood(X)
    
    def sample(self, n_samples=1, random_state=None):
        if self.mu_ is None:
            raise ValueError("Model must be fitted before sampling")
        
        if random_state is not None:
            np.random.seed(random_state)
        
        n_features = self.mu_.shape[1]
        X = np.zeros((n_samples, n_features))
        labels = np.random.choice(self.n_components, size=n_samples, p=self.pi_)
        
        for k in range(self.n_components):
            mask = labels == k
            n_k = mask.sum()
            if n_k > 0:
                X[mask] = np.random.multivariate_normal(self.mu_[k], self.cov_[k], size=n_k)
        
        return X, labels
    
    def plot_2D_model(self, ax, colors, alpha=0.3):
        from matplotlib.patches import Ellipse
        for k in range(self.n_components):
            ax.plot(self.mu_[k][0], self.mu_[k][1], 'x', color=colors[k])
            eigenvalues, eigenvectors = np.linalg.eigh(self.cov_[k])
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 2 * np.sqrt(eigenvalues) * 2
            ellipse = Ellipse(self.mu_[k], width, height, angle=angle, 
                            facecolor=colors[k], alpha=alpha, edgecolor=colors[k], linewidth=2)
            ax.add_patch(ellipse)
