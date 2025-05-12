import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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

    # Create the plot
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

    # Customize the plot
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(y_var)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_violin_ts( 
                    df: pd.DataFrame,
                    x_var: str,
                    y_var: str,
                    title: str | None = None,
                    show_mean: bool = True
                    ) -> None:
    # Set the style and figure size
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    # Create the box plot
    sns.boxplot(x=x_var, y=y_var, data=df, palette='pastel', width=0.5)

    # Create the violin plot
    sns.violinplot(x=x_var, y=y_var, data=df, scale='width', inner='quartile',
                    cut=0, color=".8")

    # Customize the plot
    plt.title(title, fontsize=15)
    plt.xlabel(x_var, fontsize=16)
    plt.ylabel(y_var, fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)

    # Add some statistics
    if show_mean:
        for i, n in enumerate(df[x_var].unique()):
            day_data = df[df[x_var] == n][y_var]
            mean = day_data.mean()
            plt.text(i, plt.ylim()[1], f'Mean: {mean:.0f}', 
                    horizontalalignment='center', fontsize=10)

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_scatter_weather(
                    df: pd.DataFrame,
                    x_var: str,
                    y_var: str,
                    hue_var: str,
                    title: str | None = None,
                    ) -> None:
    # Create the plot
    sns.scatterplot(data=df, 
                    x=x_var, 
                    y=y_var, 
                    hue=hue_var,
                    alpha=0.5)

    # Customize the plot
    plt.title(title, fontsize=16)
    plt.xlabel(x_var, fontsize=14)
    plt.ylabel(y_var, fontsize=14)

    # Customize the legend
    plt.legend(title=hue_var, labels=df[hue_var].unique(), title_fontsize=12, fontsize=10)

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_compare_two_col(
                    df: pd.DataFrame,
                    y1_var: str,
                    y2_var: str,
                    title: str | None = None,
                    ) -> None:
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=df.index, y=df[y1_var], name="y1_var"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df[y2_var], name=y2_var),
        secondary_y=True,
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Date")

    # Set y-axes titles
    fig.update_yaxes(title_text=y1_var, secondary_y=False)
    fig.update_yaxes(title_text=y2_var, secondary_y=True)

    # Update layout
    fig.update_layout(
        title_text=title,
        legend=dict(y=1, x=0.01),
        hovermode="x unified"
    )

    # Show the figure
    fig.show()

