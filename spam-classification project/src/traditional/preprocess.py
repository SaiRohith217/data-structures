import pandas as pd

def load_and_preprocess(path):
    data = pd.read_csv(path)

    print("Columns in dataset:", data.columns)  # DEBUG (see column names)

    # Case 1: Kaggle spam dataset (v1, v2)
    if 'v1' in data.columns and 'v2' in data.columns:
        data = data.rename(columns={'v1': 'label', 'v2': 'text'})

    # Case 2: Already correct names
    elif 'label' in data.columns and 'text' in data.columns:
        pass

    # Case 3: Unknown format → auto pick first 2 columns
    else:
        data = data.iloc[:, :2]
        data.columns = ['label', 'text']

    # Keep only needed columns
    data = data[['text', 'label']]

    # Convert labels to numbers
    data['label'] = data['label'].astype(str).str.lower()
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    # Drop rows with missing values
    data = data.dropna()

    return data