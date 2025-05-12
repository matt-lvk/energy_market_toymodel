import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F",
                "#00B9E3", "#619CFF", "#DB72FB"]


def plot_actual_predict(df: pd.DataFrame,
                            pred_var: str,
                            y_var: str,
                            title: str,
                        ) -> None:
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(
                data=df[pred_var], 
                marker=None,
                label=pred_var, 
                color=color_pal[3]
                )
    
    sns.scatterplot(data=df[y_var], marker='.', label=y_var, color=color_pal[1])

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