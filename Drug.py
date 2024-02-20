import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Load your dataset
df = pd.read_csv('Drug.csv')  # Replace with your actual dataset

# Assuming your target variable is 'target' and features are 'feature1', 'feature2', etc.# Replace with your actual numeric feature columns
numeric_features = ['Effective','EaseOfUse']  # Replace with your actual numeric feature columns
categorical_features = ['Drug']  # Replace with your actual categorical feature columns
target_column = 'Satisfaction'  # Replace with your actual target column

# Check if the specified feature columns exist in the DataFrame
if not set(numeric_features).issubset(df.columns):
    st.error(f"Some of the specified numeric feature columns ({numeric_features}) are not found in the dataset.")
    st.stop()

# Check if the specified target column exists in the DataFrame
if target_column not in df.columns:
    st.error(f"The specified target column ({target_column}) is not found in the dataset.")
    st.stop()

# Separate numeric and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # You may need to handle missing values
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # You may need to handle missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LinearRegression())])

# Train the model
X = df[numeric_features + categorical_features]
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Main function to create the app
def main():
    st.title('Drug Model :pill:')

    # User input for prediction
    st.sidebar.header('User Input Features')

    # Assuming the features are numeric for simplicity; you can customize for your use case
    feature1 = st.sidebar.slider('Effective', float(df['Effective'].min()), float(df['Effective'].max()), float(df['Effective'].mean()))
    feature2 = st.sidebar.slider('EaseOfUse', float(df['EaseOfUse'].min()), float(df['EaseOfUse'].max()), float(df['EaseOfUse'].mean()))
    # User input for prediction
    categorical_feature = st.sidebar.selectbox('Drug', df['Drug'].unique())  # Replace with your actual categorical feature
    user_input = pd.DataFrame({'Effective': [feature1], 'EaseOfUse': [feature2], 'Drug': [categorical_feature]})

    # Display the user input
    st.subheader('User Input:')
    st.write(user_input)

    # Make predictions
    prediction = model.predict(user_input)

    # Display prediction
    st.subheader('Prediction:')
    st.write(prediction)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display evaluation metrics
    st.subheader('Model Evaluation:')
    st.write(f'Mean Squared Error (MSE): {mse}')
    st.write(f'R-squared (R2): {r2}')

   
if __name__ == '__main__':
    main()
