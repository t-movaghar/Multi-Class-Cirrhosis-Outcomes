def process_data(df):
    import pandas as pd
    from sklearn.preprocessing import RobustScaler
    import numpy as np

    df = df.dropna()
    df = df.drop(['id'], axis=1)
    numerical_columns = df.select_dtypes(include=['number']).columns
    
    df[numerical_columns] = np.log(df[numerical_columns] + 1) #log transformation

    scaler = RobustScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns]) #scaling down
    
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    encoded_df = pd.get_dummies(df, columns=categorical_columns) #one-hot encoding
    boolean_columns = encoded_df.select_dtypes(include=bool).columns
    encoded_df[boolean_columns] = encoded_df[boolean_columns].astype(int)

    return encoded_df

def create_histogram(df):
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    num_cols = len(df.columns)
    num_rows = num_cols // 6 + (num_cols % 6 > 0) #calculating the needed amount of columns
    
    fig, axes = plt.subplots(num_rows, 6, figsize=(20, 6* num_rows))
    axes = axes.flatten()
    
    for i, column in enumerate(df.columns):
        sns.histplot(df[column], kde=True, ax=axes[i], stat='density')
        axes[i].set_title(f'Histogram of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')
    
    for j in range(num_cols, num_rows * 6): #hiding any unused subplots
        fig.delaxes(axes[j]) 
    
    plt.tight_layout()
    plt.show()