import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

class GaussianMixtureModelVectorized:
    """
    Gaussian Mixture Model fitted using the EM algorithm.
    
    Parameters
    ----------
    n_components : int, default=2
        Number of mixture components
    random_state : int or None, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, n_components=2, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        
        # Parameters (set after fitting)
        self.mu_ = None
        self.cov_ = None
        self.pi_ = None
        
        # Diagnostics (set after fitting)
        self.log_likelihoods_ = None
        self.n_iter_ = None
        self.converged_ = False
    
    def fit(self, X, max_iter=500, tol=1e-6, initial_theta=None, verbose=True):
        """
        Fit the GMM to data using the EM algorithm.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        max_iter : int, default=500
            Maximum number of EM iterations
        tol : float, default=1e-6
            Convergence tolerance (change in log-likelihood)
        initial_theta : tuple or None, default=None
            Initial parameters (mus, covs, pi)
        verbose : bool, default=True
            Show progress bar
        
        Returns
        -------
        self : object
            Returns self
        """
        self.max_iter = max_iter
        self.tol = tol

        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self._initialize_parameters(X, initial_theta)
        
        # Track log-likelihood
        self.log_likelihoods_ = []
        
        # Main EM loop
        for iteration in tqdm(range(self.max_iter), disable=not verbose):
            # E-step
            responsibilities = self._e_step(X)
            
            # M-step
            self._m_step(X, responsibilities)
            
            # Compute log-likelihood
            ll = self._compute_log_likelihood(X)
            self.log_likelihoods_.append(ll)
            
            # Check convergence
            if iteration > 0:
                ll_change = abs(self.log_likelihoods_[-1] - self.log_likelihoods_[-2])
                if ll_change < self.tol:
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
        """Initialize GMM parameters randomly."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        if theta is not None:
            self._set_parameters(*theta)
        else:
            # Initialize means by randomly selecting data points
            indices = np.random.choice(n_samples, self.n_components, replace=False)
            self._set_parameters(
                X[indices],
                np.array([np.eye(n_features) for _ in range(self.n_components)]),
                np.ones(self.n_components) / self.n_components
            )
    
    def _e_step(self, X):
        """
        E-step: Compute responsibilities.
        
        Uses log-sum-exp trick for numerical stability.
        
        Returns
        -------
        responsibilities : array, shape (n_samples, n_components)
            Posterior probabilities that each point belongs to each component
        """
        n_samples = X.shape[0]
        log_responsibilities = np.zeros((n_samples, self.n_components))
        
        # Compute log(π_k * N(x_i | μ_k, Σ_k)) for all i,k (vectorized over samples)
        for k in range(self.n_components):
            log_responsibilities[:, k] = (
                np.log(self.pi_[k]) + 
                multivariate_normal.logpdf(X, mean=self.mu_[k], cov=self.cov_[k])
            )
        
        # Log-sum-exp trick: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
        max_log = np.max(log_responsibilities, axis=1, keepdims=True)
        log_sum_exp = max_log + np.log(
            np.sum(np.exp(log_responsibilities - max_log), axis=1, keepdims=True)
        )
        log_responsibilities -= log_sum_exp
        
        return np.exp(log_responsibilities)
    
    def _m_step(self, X, responsibilities):
        """
        M-step: Update parameters using responsibilities.
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        responsibilities : array, shape (n_samples, n_components)
        """
        n_samples, n_features = X.shape
        
        # Compute N_k = sum of responsibilities for each component
        N_k = responsibilities.sum(axis=0)
        
        # Update mixing proportions
        self.pi_ = N_k / n_samples
        
        # Update means (vectorized)
        self.mu_ = (responsibilities.T @ X) / N_k[:, np.newaxis]
        
        # Update covariances
        self.cov_ = np.zeros((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = X - self.mu_[k]
            # Weighted covariance
            self.cov_[k] = (responsibilities[:, k:k+1] * diff).T @ diff / N_k[k]
            # Add regularization to prevent singularity
            self.cov_[k] += 1e-6 * np.eye(n_features)
    
    def _compute_log_likelihood(self, X):
        """
        Compute the log-likelihood of the data.
        
        Uses log-sum-exp trick for numerical stability.
        
        Returns
        -------
        log_likelihood : float
        """
        n_samples = X.shape[0]
        log_likelihood_per_point = np.zeros((n_samples, self.n_components))
        
        # Compute log(π_k * N(x_i | μ_k, Σ_k)) for all i,k
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
        """
        Predict posterior probabilities for each component.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        
        Returns
        -------
        responsibilities : array, shape (n_samples, n_components)
            Posterior probabilities
        """
        X = np.asarray(X)
        return self._e_step(X)
    
    def predict(self, X):
        """
        Predict the component labels for each point.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        
        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels (argmax of responsibilities)
        """
        responsibilities = self.predict_proba(X)
        return np.argmax(responsibilities, axis=1)
    
    def score(self, X):
        """
        Compute the log-likelihood of X under the fitted model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        
        Returns
        -------
        log_likelihood : float
        """
        X = np.asarray(X)
        return self._compute_log_likelihood(X)

    def sample(self, n_samples=1, random_state=None):
        """
        Generate random samples from the fitted GMM.
        
        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate
        random_state : int or None, default=None
            Random seed for reproducibility
        
        Returns
        -------
        X : array, shape (n_samples, n_features)
            Generated samples
        labels : array, shape (n_samples,)
            Component labels for each sample
        """
        if self.mu_ is None:
            raise ValueError("Model must be fitted before sampling")
        
        if random_state is not None:
            np.random.seed(random_state)
        
        n_features = self.mu_.shape[1]
        
        # Allocate output arrays
        X = np.zeros((n_samples, n_features))
        
        # Sample component assignments from the mixing proportions
        labels = np.random.choice(self.n_components, size=n_samples, p=self.pi_)
        
        # For each component, sample from that component
        for k in range(self.n_components):
            mask = labels == k
            n_k = mask.sum()
            if n_k > 0:
                X[mask] = np.random.multivariate_normal(self.mu_[k], self.cov_[k], size=n_k)
        
        return X, labels
    
    def plot_2D_model(self, ax, colors, alpha=0.3):
        """
        Visualize the model parameters using ellipses.
        """
        from matplotlib.patches import Ellipse
        
        for k in range(self.n_components):
            ax.plot(self.mu_[k][0], self.mu_[k][1], 'x', color=colors[k])
            
            eigenvalues, eigenvectors = np.linalg.eigh(self.cov_[k])
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 2 * np.sqrt(eigenvalues) * 2  # 2 standard deviations
            
            ellipse = Ellipse(self.mu_[k], width, height, angle=angle, 
                            facecolor=colors[k], alpha=alpha, edgecolor=colors[k], linewidth=2)
            ax.add_patch(ellipse)
