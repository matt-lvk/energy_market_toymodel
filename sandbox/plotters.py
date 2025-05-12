import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_per_column_distributions(
                            df: pd.DataFrame,
                            nGraphShown: int,
                            nGraphPerRow: int
                            ) -> None:
    """
    Plot a distribution of values in each column of df.

    Parameters
    ----------
    df : DataFrame
        DataFrame to plot
    nGraphShown : int
        Number of graphs to show
    nGraphPerRow : int
        Number of graphs per row
    
    Returns
    -------
    None
        This function doesn't return anything. It displays the plot.
    """

    nCol = df.shape[1]
    nGraphRow = (min(nCol, nGraphShown) - 1) // nGraphPerRow + 1
    
    fig, axes = plt.subplots(
                            nGraphRow, nGraphPerRow, 
                            figsize=(6 * nGraphPerRow, 8 * nGraphRow),
                            squeeze=False
                            )
    
    for i in range(min(nCol, nGraphShown)):
        row = i // nGraphPerRow
        col = i % nGraphPerRow
        ax = axes[row, col]
        
        columnDf = df.iloc[:, i]
        if not np.issubdtype(columnDf.dtype, np.number):
            columnDf.value_counts().plot.bar(ax=ax)
        else:
            columnDf.hist(ax=ax)
        
        ax.set_ylabel('counts')
        ax.set_title(f'{df.columns[i]} (column {i})')
        ax.tick_params(axis='x', rotation=90)
    
    # Remove unused subplots
    for i in range(nCol, nGraphRow * nGraphPerRow):
        row = i // nGraphPerRow
        col = i % nGraphPerRow
        fig.delaxes(axes[row, col])
    
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, graph_width: float) -> None:
    """
    Plot a graphical correlation matrix for each pair of columns in the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing the data to be plotted.
    graph_width : float
        Width of the plot in inches.

    Returns
    -------
    None
        This function doesn't return anything. It displays the plot.

    Notes
    -----
    This function creates a graphical representation of the correlation
    between each pair of columns in the input dataframe.
    """
    filename = getattr(df, 'dataframeName', 'Unnamed DataFrame')
    df = df.select_dtypes(include=[np.number]).dropna()
    df = df.loc[:, df.nunique() > 1]

    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(graph_width, graph_width), dpi=80)
    corr_mat = ax.matshow(corr)
    
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    ax.xaxis.set_ticks_position('bottom')
    
    plt.colorbar(corr_mat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.tight_layout()
    plt.show()


def plot_scatter_matrix(df: pd.DataFrame, plot_size: int, text_size: int) -> None:
    """
    Plot a scatter matrix for a given dataframe df.

    Parameters
    ----------
    df : DataFrame
        Data to plot
    plot_size : int
        Size of each subplot
    text_size : int
        Text size for the annotation
        
    Returns
    -------
    None
        This function doesn't return anything. It displays the plot.
    """
    df = df.select_dtypes(include=[np.number]).dropna()
    df = df.loc[:, df.nunique() > 1]

    if df.shape[1] > 10:
        df = df.iloc[:, :10]  # limit to first 10 columns

    fig, axes = pd.plotting.scatter_matrix(
        df,
        alpha=0.75,
        figsize=[plot_size, plot_size],
        diagonal='kde'
        )
    
    corrs = df.corr().values
    # n = len(df.columns)
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        axes[i, j].annotate(f'Corr. coef = {corrs[i, j]:.3f}',
                            (0.8, 0.2),
                            xycoords='axes fraction',
                            ha='center',
                            va='center',
                            size=text_size)
    
    plt.suptitle('Scatter and Density Plot', y=1.02)
    plt.tight_layout()
    plt.show()