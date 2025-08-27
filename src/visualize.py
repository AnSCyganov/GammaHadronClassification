import matplotlib.pyplot as plt

def visualize(df):
    for label in df.columns[:-1]:
        plt.hist(df[df['class']==1][label], color='blue', label='gamma', alpha=0.5, density=True)
        plt.hist(df[df['class']==0][label], color='red', label='beta', alpha=0.5, density=True)
        plt.title(label)
        plt.ylabel('Probability')
        plt.xlabel(label)
        plt.legend()
        plt.show()