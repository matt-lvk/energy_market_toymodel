import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F",
                "#00B9E3", "#619CFF", "#DB72FB"]


def plot_actual_predict(
                        df: pd.DataFrame,
                        y_var: str,
                        pred_var: str,
                        title: str,
                        ) -> None:
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    plt.figure(figsize=(12, 6))
    
    sns.lineplot(
                data=df[y_var], 
                marker=None,
                label=y_var, 
                color='grey'
                )
    
    sns.lineplot(
                data=df[pred_var], 
                marker=None,
                label=pred_var, 
                color=color_pal[0]
                )
    
    # sns.scatterplot(
    #                 data=df[y_var],
    #                 marker='.',
    #                 label=y_var,
    #                 color=color_pal[1]
    #                 )

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(y_var)

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def plot_violin_ts( 
                    df: pd.DataFrame,
                    x_var: str,
                    y_var: str,
                    title: str | None = None,
                    show_mean: bool = True
                    ) -> None:

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    # box plot
    sns.boxplot(x=x_var, y=y_var, data=df, palette='pastel', width=0.5)

    # violin plot
    sns.violinplot(x=x_var, y=y_var, data=df, scale='width', inner='quartile',
                    cut=0, color=".8")

    plt.title(title, fontsize=15)
    plt.xlabel(x_var, fontsize=16)
    plt.ylabel(y_var, fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)

    # Add mean in pic
    if show_mean:
        for i, n in enumerate(df[x_var].unique()):
            day_data = df[df[x_var] == n][y_var]
            mean = day_data.mean()
            plt.text(i, plt.ylim()[1], f'Mean: {mean:.0f}', 
                    horizontalalignment='center', fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_scatter_weather(
                    df: pd.DataFrame,
                    x_var: str,
                    y_var: str,
                    hue_var: str,
                    title: str | None = None,
                    ) -> None:

    sns.scatterplot(data=df, 
                    x=x_var, 
                    y=y_var, 
                    hue=hue_var,
                    alpha=0.5)

    plt.title(title, fontsize=16)
    plt.xlabel(x_var, fontsize=14)
    plt.ylabel(y_var, fontsize=14)

    plt.legend(title=hue_var, labels=df[hue_var].unique(), title_fontsize=12, fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_compare_two_col(
                    df: pd.DataFrame,
                    y1_var: str,
                    y2_var: str,
                    title: str | None = None,
                    ) -> None:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=df.index, y=df[y1_var], name="y1_var"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df[y2_var], name=y2_var),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Date")

    fig.update_yaxes(title_text=y1_var, secondary_y=False)
    fig.update_yaxes(title_text=y2_var, secondary_y=True)

    fig.update_layout(
        title_text=title,
        legend=dict(y=1, x=0.01),
        hovermode="x unified"
    )

    fig.show()


def plotly_actual_predict(
                        df: pd.DataFrame,
                        y_var: str,
                        pred_var: str,
                        title: str,
                        split_point: datetime | None = None,
                        start_train_date: datetime | None = None,
                        ) -> None:
    """
    Show the plot actual and predicted values of a time series.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the actual and predicted values.
    y_var : str
        Name of the column containing the actual values.
    pred_var : str
        Name of the column containing the predicted values.
    title : str
        Title of the plot.
    
    Returns
    -------
    None
    """
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df[y_var], name=y_var, line=dict(color='grey'))
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df[pred_var], name=pred_var, line=dict(color='blue'))
    )
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=y_var,
        legend_title='Legend',
        font=dict(size=12),
        hovermode="x unified"
    )

    if split_point:
        fig.add_vline(x=split_point, line_width=3, line_dash="dash", line_color="green")
    
    if start_train_date:
        fig.add_vline(x=start_train_date, line_width=3, line_color="orange")
    
    # add nice button 
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    fig.show()

