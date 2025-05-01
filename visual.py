import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os
import argparse

def analyze_csv(file_path, output_dir="analysis_results"):
    """
    Perform comprehensive visual analysis on each feature in a CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    output_dir : str
        Directory to save the visualization results
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the CSV file
    print(f"Reading CSV file: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        # Try with different encodings if the default fails
        try:
            df = pd.read_csv(file_path, encoding='latin1')
            print("Successfully read file with latin1 encoding")
        except:
            try:
                df = pd.read_csv(file_path, encoding='utf-16')
                print("Successfully read file with utf-16 encoding")
            except Exception as e:
                print(f"Failed to read the file: {e}")
                return
    
    # Print basic information about the dataset
    print("\n===== DATASET SUMMARY =====")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nColumn names:")
    for col in df.columns:
        print(f"- {col}")
    
    print("\n===== DATA TYPES =====")
    print(df.dtypes)
    
    print("\n===== MISSING VALUES =====")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_data = pd.DataFrame({'Missing Values': missing, 'Percentage': missing_percent})
    print(missing_data[missing_data['Missing Values'] > 0])
    
    # Save missing values chart
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(missing)), missing)
    plt.xticks(range(len(missing)), missing.index, rotation=90)
    plt.title('Missing Values by Feature')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/missing_values.png")
    plt.close()
    
    # Analyze each column
    print("\n===== COLUMN ANALYSIS =====")
    
    # Create correlation matrix for numeric variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:  # Only create correlation matrix if we have more than one numeric column
        plt.figure(figsize=(12, 10))
        corr = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                   square=True, linewidths=.5)
        plt.title('Correlation Matrix of Numeric Features')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_matrix.png")
        plt.close()
        print("\nCreated correlation matrix for numeric features")
    
    # Analyze each column individually
    for column in df.columns:
        print(f"\nAnalyzing column: {column}")
        
        # Check data type
        dtype = df[column].dtype
        print(f"Data type: {dtype}")
        
        # Handle different data types
        if pd.api.types.is_numeric_dtype(df[column]):
            # Numeric column analysis
            stats = df[column].describe()
            print(stats)
            
            # Distribution plot
            plt.figure(figsize=(12, 6))
            
            # Create subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Histogram
            sns.histplot(df[column].dropna(), kde=True, ax=ax1)
            ax1.set_title(f'Distribution of {column}')
            
            # Box plot
            sns.boxplot(y=df[column].dropna(), ax=ax2)
            ax2.set_title(f'Box Plot of {column}')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{column}_distribution.png")
            plt.close()
            
            # Identify outliers
            Q1 = stats['25%']
            Q3 = stats['75%']
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
            if len(outliers) > 0:
                print(f"Potential outliers found: {len(outliers)} values")
                print(f"Outlier range: < {lower_bound} or > {upper_bound}")
        
        elif pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            # Categorical/String column analysis
            value_counts = df[column].value_counts()
            print(f"Unique values: {df[column].nunique()}")
            
            if df[column].nunique() <= 20:  # Only show distribution for columns with reasonable number of categories
                print("Value distribution:")
                print(value_counts.head(10))  # Show top 10 values
                
                # Bar plot for categorical data
                plt.figure(figsize=(12, 6))
                value_counts_limited = value_counts.head(15)  # Limit to top 15 categories for readability
                sns.barplot(x=value_counts_limited.index, y=value_counts_limited.values)
                plt.title(f'Distribution of {column} (Top 15 categories)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/{column}_distribution.png")
                plt.close()
                
                # If fewer than 10 categories, create pie chart
                if df[column].nunique() <= 10:
                    plt.figure(figsize=(10, 10))
                    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
                    plt.title(f'Distribution of {column}')
                    plt.axis('equal')
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/{column}_pie_chart.png")
                    plt.close()
            else:
                print(f"Too many unique values ({df[column].nunique()}) to display. Top 10:")
                print(value_counts.head(10))
                
                # Create a word cloud for text data with many unique values
                try:
                    from wordcloud import WordCloud
                    text = ' '.join(df[column].dropna().astype(str))
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                    
                    plt.figure(figsize=(12, 6))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.title(f'Word Cloud for {column}')
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/{column}_wordcloud.png")
                    plt.close()
                except ImportError:
                    print("WordCloud package not installed. Skipping word cloud visualization.")
        
        elif pd.api.types.is_datetime64_dtype(df[column]):
            # Date/time column analysis
            print(f"Earliest date: {df[column].min()}")
            print(f"Latest date: {df[column].max()}")
            print(f"Date range: {df[column].max() - df[column].min()}")
            
            # Time series plot
            plt.figure(figsize=(12, 6))
            df[column].value_counts().sort_index().plot()
            plt.title(f'Time Series Distribution of {column}')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{column}_timeseries.png")
            plt.close()
        
        else:
            # Other data types
            print(f"Unable to analyze column {column} with dtype {dtype}")
    
    # If there are at least two numeric columns, create pair plots for them
    if len(numeric_cols) >= 2:
        sample_size = min(1000, len(df))  # Limit sample size for pair plots to prevent excessive computation
        try:
            # Sample the dataframe if it's large
            sampled_df = df.sample(sample_size) if len(df) > sample_size else df
            
            # Split numeric columns into groups of 5 to avoid creating too large pairplots
            for i in range(0, len(numeric_cols), 5):
                cols_group = numeric_cols[i:i+5]
                if len(cols_group) > 1:  # Only create pairplot if there are at least 2 columns
                    plt.figure(figsize=(15, 15))
                    sns.pairplot(sampled_df[cols_group], diag_kind='kde')
                    plt.suptitle(f'Pair Plot of Numeric Features (Group {i//5 + 1})', y=1.02)
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/pairplot_group_{i//5 + 1}.png")
                    plt.close()
                    print(f"Created pair plot for numeric features group {i//5 + 1}")
        except Exception as e:
            print(f"Error creating pair plots: {e}")
    
    # Create a feature relationship analysis for selected categorical and numeric columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols and numeric_cols:
        try:
            # Select a limited number of combinations for visualization
            for cat_col in cat_cols[:3]:  # Take up to 3 categorical columns
                if df[cat_col].nunique() <= 10:  # Only use categorical cols with limited unique values
                    for num_col in numeric_cols[:3]:  # Take up to 3 numeric columns
                        plt.figure(figsize=(12, 6))
                        sns.boxplot(x=cat_col, y=num_col, data=df)
                        plt.title(f'Relationship: {cat_col} vs {num_col}')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        plt.savefig(f"{output_dir}/{cat_col}_vs_{num_col}_boxplot.png")
                        plt.close()
                        print(f"Created relationship plot for {cat_col} vs {num_col}")
        except Exception as e:
            print(f"Error creating relationship plots: {e}")
    
    print(f"\nAnalysis complete! Results saved to {output_dir} directory.")

if __name__ == "__main__":
    # Simple input prompt for CSV file path
    file_path = input("Enter the path to your CSV file: ")
    output_dir = "analysis_results"
    
    # Remove quotes if user accidentally included them
    file_path = file_path.strip('"').strip("'")
    
    print(f"\nAnalyzing file: {file_path}")
    analyze_csv(file_path, output_dir)