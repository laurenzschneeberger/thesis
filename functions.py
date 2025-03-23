import pandas as pd 
import numpy as np
import statsmodels.api as sm
from factor_analyzer.rotator import Rotator
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from estimators import pca



def random_subsample(datasets, correlation_matrices, n_samples=30, seed=42):
    np.random.seed(seed)
    
    new_dfs = []
    new_corrs = []
    
    for df, corr in zip(datasets, correlation_matrices):
        tickers = np.random.choice(df.columns, size=n_samples, replace=False)
        new_dfs.append(df[tickers])
        new_corrs.append(corr.loc[tickers, tickers])
    
    return new_dfs, new_corrs

def random_subsample2(datasets, n_samples=30, seed=42):
    np.random.seed(seed)
    
    new_dfs = []
    new_corrs = []
    
    for df in datasets:
        tickers = np.random.choice(df.columns, size=n_samples, replace=False)
        new_dfs.append(df[tickers])
    
    return new_dfs

def calculate_factor_returns(returns, betas):
    """
    Given knowledge of the betas, run a x-sectional regression to estimate the factor returns. 
    IN: a betas dataframe with colnames starting at factor_0 etc, and a returns dataframe with tickers in the columns.
    OUT: a factor return matrix 
    """

    X = pd.DataFrame(betas, index=returns.columns, 
                    columns=[f'factor_{i}' for i in range(betas.shape[1])])
    
    factor_returns = pd.DataFrame(index=returns.index,
                                columns=[f'factor_{i}' for i in range(betas.shape[1])])
    
    for t in returns.index:
        y = returns.loc[t]
        
        model = sm.OLS(y, X)
        results = model.fit()
        
        factor_returns.loc[t] = results.params
    
    factor_returns.columns = [f'factor_{i}' for i in range(len(betas.columns))]
    
    return factor_returns

def rotate_oblimin(betas):
  
  rotator = Rotator(method='oblimin')

  rotated_loadings = rotator.fit_transform(betas)
  
  # Get factor correlations (phi matrix)
  factor_correlations = rotator.phi_
  
  # Calculate unsorted eigenvalues (sum of squared loadings per factor)
  eigenvalues = np.sum(rotated_loadings**2, axis=0)
  
  # Sort factors by eigenvalue magnitude (descending)
  sort_idx = np.argsort(-eigenvalues)
  eigenvalues = eigenvalues[sort_idx]
  rotated_loadings = rotated_loadings[:, sort_idx]
  factor_correlations = factor_correlations[sort_idx][:, sort_idx]
  
  structure = rotated_loadings.dot(factor_correlations)
  
  communalities = np.sum(rotated_loadings * structure, axis=1)
  
  var_ratio = communalities.sum() / len(betas.index)
  
  factor_explained = np.sum(rotated_loadings * structure, axis=0) / len(betas.index)
  
  rotated_loadings_df = pd.DataFrame(
      rotated_loadings,
      index=betas.index,
      columns=[f'factor_{i}' for i in range(rotated_loadings.shape[1])]
  )
  
  return factor_explained, var_ratio, eigenvalues, communalities, factor_correlations, rotated_loadings_df

def rotate_varimax(betas):
    rotator = Rotator(method='varimax')
    rotated_loadings = rotator.fit_transform(betas)
    
    # For varimax (orthogonal rotation), phi is the identity matrix.
    n_factors = rotated_loadings.shape[1]
    factor_correlations = np.eye(n_factors)
    
    # Calculate unsorted eigenvalues (sum of squared loadings per factor)
    eigenvalues = np.sum(rotated_loadings**2, axis=0)
    
    # Sort factors by eigenvalue magnitude (descending)
    sort_idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    rotated_loadings = rotated_loadings[:, sort_idx]
    factor_correlations = factor_correlations[sort_idx][:, sort_idx]
    
    # With orthogonal rotations, structure equals rotated loadings.
    structure = rotated_loadings
    communalities = np.sum(rotated_loadings**2, axis=1)
    
    var_ratio = communalities.sum() / len(betas.index)
    factor_explained = np.sum(rotated_loadings**2, axis=0) / len(betas.index)
    
    rotated_loadings_df = pd.DataFrame(
        rotated_loadings,
        index=betas.index,
        columns=[f'factor_{i}' for i in range(rotated_loadings.shape[1])]
    )
    
    return factor_explained, var_ratio, eigenvalues, communalities, factor_correlations, rotated_loadings_df

def rotate_promax(betas):
  
  rotator = Rotator(method='promax')

  rotated_loadings = rotator.fit_transform(betas)
  
  # Get factor correlations (phi matrix)
  factor_correlations = rotator.phi_
  
  # Calculate unsorted eigenvalues (sum of squared loadings per factor)
  eigenvalues = np.sum(rotated_loadings**2, axis=0)
  
  # Sort factors by eigenvalue magnitude (descending)
  sort_idx = np.argsort(-eigenvalues)
  eigenvalues = eigenvalues[sort_idx]
  rotated_loadings = rotated_loadings[:, sort_idx]
  factor_correlations = factor_correlations[sort_idx][:, sort_idx]
  
  structure = rotated_loadings.dot(factor_correlations)
  
  communalities = np.sum(rotated_loadings * structure, axis=1)
  
  var_ratio = communalities.sum() / len(betas.index)
  
  factor_explained = np.sum(rotated_loadings * structure, axis=0) / len(betas.index)
  
  rotated_loadings_df = pd.DataFrame(
      rotated_loadings,
      index=betas.index,
      columns=[f'factor_{i}' for i in range(rotated_loadings.shape[1])]
  )
  
  return factor_explained, var_ratio, eigenvalues, communalities, factor_correlations, rotated_loadings_df

def correlationplot(datasets):
    """
    Plots correlation matrices for factors using different rotation methods.
    IN: list of four beta matrices e.g. [betas00, betas05, betas10, betas15]
    OUT: a plot
    """
    
    rotation_functions = [rotate_varimax, rotate_oblimin, rotate_promax]
    titles = ['2000-2004', '2005-2009', '2010-2014', '2015-2019']
    rotation_names = ['Varimax', 'Oblimin', 'Promax']
    fig, axes = plt.subplots(3, 4, figsize=(15, 12))
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    cmap = 'RdBu_r'

    for i, rotation_function in enumerate(rotation_functions):
        for j, (df, title) in enumerate(zip(datasets, titles)):
            ax = axes[i, j]

            factor_correlations = rotation_function(df)[4]
            
            # Convert to numpy array if it's a DataFrame
            if hasattr(factor_correlations, 'values'):
                factor_correlations = factor_correlations.values
            
            # Get dimensions of the correlation matrix
            n_factors = factor_correlations.shape[0]

            im = ax.imshow(factor_correlations, cmap=cmap, vmin=-1, vmax=1)

            ax.set_title(f'{title}', fontsize=12)

            # Dynamically set ticks based on matrix dimensions
            ax.set_xticks(range(n_factors))
            ax.set_xticklabels(range(1, n_factors + 1))
            ax.set_yticks(range(n_factors))
            ax.set_yticklabels(range(1, n_factors + 1))

            ax.grid(False)

            # Plot correlation values in lower triangle
            for row in range(n_factors):
                for col in range(n_factors):
                    if row > col:  # Only lower triangle
                        text = ax.text(
                            col,
                            row,
                            f'{factor_correlations[row][col]:.2f}',
                            ha='center',
                            va='center',
                            color="black",
                            fontsize=12,
                        )

        axes[i, 0].set_ylabel(f'{rotation_names[i]}', size=12)

    plt.tight_layout()
    plt.show()

def rotation_variance_plot(datasets): 
    for betas in datasets: 
        n_factors = betas.shape[1]
        
        # --- Pre-rotation variances
        explained_raw = (betas**2).sum()/betas.shape[0]
        cumulative_raw = np.cumsum(explained_raw)

        x_ticks = range(n_factors)
        offset = 0.1
        delim_width = 0.05

        # --- Plot
        plt.figure(figsize=(12, 12))
        fig, (ax1, ax2, ax3) = plt.subplots(3, 2, figsize=(12, 12))

        def style_subplots(ax_left, ax_right, title):
            fig.add_subplot(111, frame_on=False)
            plt.tick_params(labelcolor="none", bottom=False, left=False)
            
            for ax in [ax_left, ax_right]:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(0.5)
                ax.spines['bottom'].set_linewidth(0.5)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels([str(i+1) for i in x_ticks])
                ax.tick_params(axis='both', which='major', labelsize=9)
                ax.set_xlabel('Number of Factors', fontsize=9)
            
            ax_left.set_ylim(0, max(explained_raw)*1.1)
            ax_left.set_ylabel('Explained Variance', fontsize=9)
            ax_left.legend(frameon=False, fontsize=9)

            max_cumulative = max(cumulative_raw.max(), cumulative_rotated.max())
            ax_right.set_ylim(0, max_cumulative * 1.1)
            ytick_max = np.ceil(max_cumulative * 10) / 10
            ax_right.set_yticks(np.arange(0, ytick_max + 0.1, 0.1))
            ax_right.set_ylabel('Cumulative Explained Variance', fontsize=9)
            
        for idx, (axes, rotation_func, title) in enumerate(zip(
            [(ax1[0], ax1[1]), (ax2[0], ax2[1]), (ax3[0], ax3[1])],
            [rotate_varimax, rotate_oblimin, rotate_promax],
            ['Varimax', 'Oblimin', 'Promax']
        )):
            ax_left, ax_right = axes
            x_ticks_raw = [x - offset for x in x_ticks]
            x_ticks_rotated = [x + offset for x in x_ticks]
            
            # Left subplot - bar chart
            for x, y in zip(x_ticks_raw, explained_raw):
                ax_left.vlines(x, 0, y, color='#404040', linewidth=0.5, 
                            label='Raw' if x == x_ticks_raw[0] else "")
                ax_left.hlines(y, x-delim_width, x+delim_width, color='black', linewidth=1)

            # Rotated bars
            rotated_variance = rotation_func(betas)[0]
            for x, y in zip(x_ticks_rotated, rotated_variance):
                ax_left.vlines(x, 0, y, color='#404040', linestyle='--', linewidth=0.5, 
                            label=title if x == x_ticks_rotated[0] else "")
                ax_left.hlines(y, x-delim_width, x+delim_width, color='black', linewidth=1)

            # Right subplot - cumulative explained variance
            x_axis = range(n_factors)
            ax_right.plot(x_axis, cumulative_raw, linewidth=0.5, color='#404040')
            
            # Calculate cumulative variance for rotated solution
            cumulative_rotated = np.cumsum(rotated_variance)
            ax_right.plot(x_axis, cumulative_rotated, linewidth=0.5, color='#404040', 
                        linestyle='--')
            
            # Place markers at the end points
            last_idx = n_factors - 1
            ax_right.scatter(last_idx, cumulative_raw[last_idx], color='black', s=20, zorder=3)
            ax_right.scatter(last_idx, cumulative_rotated[last_idx], color='black', s=20, zorder=3)

            # Style subplots
            style_subplots(ax_left, ax_right, title)

        plt.tight_layout()
        plt.show()


def plot_factor_returns(factor_returns_list):
    """
    Plot factor returns across different time periods in a grid layout.
    
    Parameters:
    factor_returns_list: List of pandas DataFrames containing factor returns
    """
    # Validate input length
    if len(factor_returns_list) != 4:
        raise ValueError("Must provide exactly 4 DataFrames")
    
    # Clean and prepare datasets
    datasets = {
        i: df.replace([np.inf, -np.inf], np.nan).dropna() 
        for i, df in enumerate(factor_returns_list)
    }
    
    # Determine number of factors (rows) based on maximum columns in any dataset
    n_factors = max(df.shape[1] for df in datasets.values())
    
    # Create subplot grid
    fig, axes = plt.subplots(n_factors, 4, figsize=(6, 1.6 * n_factors))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    
    def style_axes(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
    
    def calculate_cumulative_returns(returns):
        if len(returns) == 0:
            return pd.Series([100])
        std_returns = (returns - returns.mean()) / returns.std()
        cum_returns = std_returns.cumsum()
        return 100 + cum_returns
    
    # Calculate global min/max
    global_min, global_max = float('inf'), float('-inf')
    for i in range(n_factors):
        for j in range(4):
            if f'factor_{i}' in datasets[j].columns:
                cum_returns = calculate_cumulative_returns(datasets[j][f'factor_{i}'])
                if not cum_returns.empty:
                    global_min = min(global_min, cum_returns.min())
                    global_max = max(global_max, cum_returns.max())
    
    # Handle case where no valid data was found
    if global_min == float('inf') or global_max == float('-inf'):
        global_min, global_max = 90, 110
    
    # Add padding to global y-limits
    y_range = global_max - global_min
    y_pad = max(y_range * 0.1, 1)
    global_y_min, global_y_max = global_min - y_pad, global_max + y_pad
    
    # Plot each factor-period combination
    for i in range(n_factors):
        for j in range(4):
            df = datasets[j]
            if f'factor_{i}' in df.columns:
                cum_returns = calculate_cumulative_returns(df[f'factor_{i}'])
                axes[i,j].plot(cum_returns.index, cum_returns.values, 
                             linewidth=0.5, color='#404040')
            else:
                axes[i,j].set_yticks([])
                axes[i,j].text(0.5, 0.5, 'Not Applicable', 
                             horizontalalignment='center',
                             verticalalignment='center',
                             transform=axes[i,j].transAxes,
                             fontsize=8)
            
            axes[i,j].grid(False)
            style_axes(axes[i,j])
            axes[i,j].set_ylim(global_y_min, global_y_max)
            
            # Set titles and labels
            if i == 0:
                axes[i,j].set_title(f'20{str(j*5).zfill(2)}-20{str(j*5+4).zfill(2)}', 
                                  fontsize=8)
            if j == 0:
                axes[i,j].set_ylabel(f'Factor {i+1}', fontsize=8)
            elif f'factor_{i}' in df.columns:
                axes[i,j].set_yticks([100])
                axes[i,j].set_yticklabels([])
            
            axes[i,j].set_xticks([])
            axes[i,j].tick_params(axis='y', which='major', labelsize=8)
    
    plt.tight_layout()
    return fig, axes

def mcp(matrix): 
    '''
    MCP with logged and non-logged eigenvalues

    IN: Covariance Matrix
    OUT: (Retained k non-logged, retained k logged)
    '''
    
    # We maximize the first difference of (log) eigenvalues
    diff = (np.diff(pca(matrix)[1]))[1:]
    log_diff = (np.diff(np.log(pca(matrix)[1])))[1:]
    
    
    return (np.argmax(abs(diff))+1, np.argmax(abs(log_diff))+1)

def aet(cov_matrix, return_matrix): 

    # Threshold
    n = cov_matrix.columns.nunique()
    T = return_matrix.shape[0]
    gamma = n / T

    # Computation
    retained = sum(np.linalg.eigh(cov_matrix)[0] > 1 + np.sqrt(gamma))

    return retained, gamma

def kaiser(correlation_matrix): 
    significant = sum(np.linalg.eigh(correlation_matrix)[0] > 1)
    return significant

def ic_factors(corr_matrix, dataset, k_max=None):
    """
    Implements Bai and Ng's information criterion for factor selection
    
    Args:
        corr_matrix: np.array, correlation matrix
        n_samples: int, number of samples
        k_max: int, optional maximum number of factors to consider
    
    Returns:
        int: optimal number of factors
    """
    
    # Get eigenvalues in descending order
    evals = np.linalg.eigh(corr_matrix)[0][::-1]
    
    # Shape of dataset
    p = len(evals)
    n = dataset.shape[0] # number of samples
    
    if k_max is None:
        k_max = min(n//3, p//3)
    
    # Calculate g(n,p)
    def g(n, p):
        return ((n + p)/(n * p)) * np.log(n * p/(n + p))
    
    # Estimate sigma^2 (using mean of smallest eigenvalues as proxy)
    sigma2_hat = np.mean(evals[k_max:])
    
    # Calculate V(k) and PC(k) for each k
    pc_scores = []
    for k in range(k_max + 1):
        # V(k) = p^(-1) * sum of eigenvalues beyond k
        v_k = np.mean(evals[k:])
        
        # PC(k) = V(k) + k * sigma2_hat * g(n,p)
        pc_k = v_k + k * sigma2_hat * g(n, p)
        pc_scores.append(pc_k)
    
    # Return k that minimizes PC(k)
    return np.argmin(pc_scores)

def scree(correlation_matrices, cutoffs, xlim=None): 
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.grid': False,
        'xtick.color': 'black',
        'ytick.color': 'black',
        'text.color': 'black',
        'font.family': ['sans-serif'],
        'font.sans-serif': ['Arial']
    })

    plot_params = {
        'color': '#000000',
        'marker': None,
        'linestyle': '-',
        'linewidth': 0.5
    }

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))

    # Use xlim if provided, otherwise use full range
    x = range(1, correlation_matrices[0].shape[0]+1)
    if xlim:
        x = range(1, xlim+1)

    # Eigenvalues to plot 
    evals_00 = np.linalg.eigh(correlation_matrices[0])[0][::-1]
    evals_05 = np.linalg.eigh(correlation_matrices[1])[0][::-1]
    evals_10 = np.linalg.eigh(correlation_matrices[2])[0][::-1]
    evals_15 = np.linalg.eigh(correlation_matrices[3])[0][::-1]

    # Plot lines without markers
    line1, = ax1.plot(x, evals_00[:xlim] if xlim else evals_00, **plot_params)
    line2, = ax2.plot(x, evals_05[:xlim] if xlim else evals_05, **plot_params)
    line3, = ax3.plot(x, evals_10[:xlim] if xlim else evals_10, **plot_params)
    line4, = ax4.plot(x, evals_15[:xlim] if xlim else evals_15, **plot_params)

    # Add specific markers
    marker_params = {'color': '#000000', 'marker': '.', 'markersize': 5}
    ax1.plot(cutoffs[0], evals_00[cutoffs[0]-1], **marker_params)
    ax2.plot(cutoffs[1], evals_05[cutoffs[1]-1], **marker_params)
    ax3.plot(cutoffs[2], evals_10[cutoffs[2]-1], **marker_params)
    ax4.plot(cutoffs[3], evals_15[cutoffs[3]-1], **marker_params)

    # Configure axes with titles and cutoff labels
    axes = [ax1, ax2, ax3, ax4]
    titles = ['2000-2004', '2005-2009', '2010-2014', '2015-2019']

    for ax, title, cutoff in zip(axes, titles, cutoffs):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out', length=2, width=0.5)
        ax.set_title(title, pad=10)
        
        # Set x-axis limits and ticks with padding on left
        if xlim:
            ax.set_xlim(0, xlim)  # Start at 0 to create padding
            ax.set_xticks([cutoff])  # Only one tick at the cutoff value
        else:
            ax.set_xlim(0, len(x))  # Start at 0 to create padding
            ax.set_xticks([cutoff])  # Only one tick at the cutoff value
            
        ax.set_xticklabels([str(cutoff)])
        
        if ax != ax1:
            ax.set_yticklabels([])

    ax1.set_ylabel('Eigenvalue')
    plt.tight_layout()
    plt.rcdefaults()


def log_scree(correlation_matrices, cutoffs):
    """
    Create log scree plots for multiple correlation matrices.

    Parameters:
    correlation_matrices : list of np.arrays
        List of 4 correlation matrices to analyze
    cutoffs : list of int
        List of 4 cutoff points to mark with dots
    """
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.grid': False,
        'xtick.color': 'black',
        'ytick.color': 'black',
        'text.color': 'black',
        'font.family': ['sans-serif'],
        'font.sans-serif': ['Arial']
    })

    plot_params = {
        'color': '#000000',
        'marker': '.',
        'markersize': 0,
        'linestyle': '-',
        'linewidth': 0.5
    }

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))

    # Calculate log eigenvalues for all matrices
    log_evals = [np.log(np.linalg.eigh(matrix)[0][::-1]) for matrix in correlation_matrices]

    # Find global min and max for consistent y-axis limits
    min_val = min(np.min(evals) for evals in log_evals)
    max_val = max(np.max(evals) for evals in log_evals)

    # Plot for each matrix
    axes = [ax1, ax2, ax3, ax4]
    titles = ['2000-2004', '2005-2009', '2010-2014', '2015-2019']

    for ax, evals, cutoff, title in zip(axes, log_evals, cutoffs, titles):
        # Plot full eigenvalue spectrum
        x = range(1, len(evals) + 1)
        ax.plot(x, evals, zorder=2, **plot_params)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='gray', linewidth=0.5, zorder=1, alpha=0.5)
        
        # Add black dot at cutoff
        ax.plot(cutoff, evals[cutoff-1], 'o', color='black', markersize=4, zorder=3)
        
        # Configure axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out', length=2, width=0.5)
        ax.set_title(title, pad=10)
        ax.set_ylim(-2, 2)
        
        # Show only the cutoff value on x-axis
        ax.set_xticks([cutoff])
        ax.set_xticklabels([str(cutoff)])
        
        # Remove y-axis labels except for first plot
        if ax != ax1:
            ax.set_yticklabels([])

    ax1.set_ylabel('Log Eigenvalue')
    plt.tight_layout()
    plt.rcdefaults()

def plot_rotated_variances_apca(corr, n_factors):
    """
    Creates a (3,2) subplot comparing unrotated vs rotated factor analyses
    with different rotation methods.
    """
    fig, ((ax1_left, ax1_right), 
          (ax2_left, ax2_right), 
          (ax3_left, ax3_right)) = plt.subplots(3, 2, figsize=(12, 12))
    
    # Calculate unrotated variances
    eigenvalues, _ = np.linalg.eigh(corr)
    eigenvalues = eigenvalues[~np.isnan(eigenvalues)]
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    total_variance = np.trace(corr)
    explained_raw = eigenvalues / total_variance
    cumulative_raw = np.cumsum(explained_raw)
    
    x_ticks = range(n_factors)
    offset = 0.1
    delim_width = 0.05
    
    def style_subplots(ax_left, ax_right, max_cumulative):
        for ax in [ax_left, ax_right]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([str(i+1) for i in x_ticks])
            ax.tick_params(axis='both', which='major', labelsize=9)
            ax.set_xlabel('Number of Factors', fontsize=9)
        
        ax_left.set_ylabel('Explained Variance', fontsize=9)
        ax_right.set_ylabel('Cumulative Explained Variance', fontsize=9)
        ax_left.legend(frameon=False, fontsize=9)
        
        # Dynamic y-axis for right subplot
        y_max = np.ceil(max_cumulative * 10) / 10
        ax_right.set_ylim(0, y_max + 0.1)
        ax_right.set_yticks(np.arange(0, y_max + 0.1, 0.1))
    
    def get_rotated_variance(loadings, factor_corr=None):
        """Calculate variance explained by rotated factors"""
        if factor_corr is None:  # Orthogonal rotation
            return np.sum(loadings**2, axis=0) / total_variance
        else:  # Oblique rotation
            return np.sum(loadings**2, axis=0) / total_variance
    
    def plot_rotation(ax_left, ax_right, rotation_method):
        x_ticks_raw = [x - offset for x in x_ticks]
        x_ticks_rotated = [x + offset for x in x_ticks]
        
        # Left subplot - unrotated bars
        max_height = 0
        
        for x, y in zip(x_ticks_raw, explained_raw[:n_factors]):
            ax_left.vlines(x, 0, y, color='#404040', linewidth=0.5,
                        label='Raw' if x == x_ticks_raw[0] else "")
            ax_left.hlines(y, x-delim_width, x+delim_width, color='black', linewidth=1)
            max_height = max(max_height, y)
        
        # Calculate rotated variances
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation_method)
        fa.fit(corr)
        
        if rotation_method in ['oblimin', 'promax']:
            factor_corr = fa.corr_
        else:
            factor_corr = None
            
        rotated_variance = get_rotated_variance(fa.loadings_, factor_corr)
        scaling_factor = np.sum(explained_raw[:n_factors]) / np.sum(rotated_variance)
        rotated_variance = rotated_variance * scaling_factor
        cumulative_rotated = np.cumsum(rotated_variance)
        
        # Rotated bars
        for x, y in zip(x_ticks_rotated, rotated_variance):
            ax_left.vlines(x, 0, y, color='#404040', linestyle='--', linewidth=0.5,
                        label=rotation_method.capitalize() if x == x_ticks_rotated[0] else "")
            ax_left.hlines(y, x-delim_width, x+delim_width, color='black', linewidth=1)
            max_height = max(max_height, y)
    
        ax_left.set_ylim(0, max_height * 1.1)
        
        # Right subplot - cumulative plots
        ax_right.plot(x_ticks, cumulative_raw[:n_factors], linewidth=0.5, color='#404040')
        ax_right.plot(x_ticks, cumulative_rotated, linewidth=0.5, 
                    color='#404040', linestyle='--')
        
        # Place marker at the last point of both lines
        last_idx = n_factors - 1
        ax_right.scatter(last_idx, cumulative_raw[last_idx], color='black', s=20, zorder=3)
        ax_right.scatter(last_idx, cumulative_rotated[last_idx], color='black', s=20, zorder=3)
        
        max_cumulative = max(max(cumulative_raw[:n_factors]), max(cumulative_rotated))
        style_subplots(ax_left, ax_right, max_cumulative)
    
    # Plot each rotation method
    plot_rotation(ax1_left, ax1_right, 'varimax')
    plot_rotation(ax2_left, ax2_right, 'oblimin')
    plot_rotation(ax3_left, ax3_right, 'promax')
    
    plt.tight_layout()
    return fig
