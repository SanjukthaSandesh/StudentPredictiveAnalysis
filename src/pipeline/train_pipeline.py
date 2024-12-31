import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from src.utils import save_object

def train_model():
    try:
        # Load dataset
        data_path = r'C:\Users\sanju\Downloads\student\Predicting-Student-Performance-Using-Machine-Learning-main\Notebook\data\stud_final.csv'
        df = pd.read_csv(data_path)

        # Define features and target
        X = df.drop(columns=["CGPA"])  # Target column
        y = df["CGPA"]

        # Define preprocessing
        numeric_features = ['CGPA100', 'CGPA200', 'CGPA300', 'CGPA400', 'SGPA']
        categorical_features = ['Gender', 'Prog Code']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(), categorical_features)
            ]
        )

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor())
        ])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        pipeline.fit(X_train, y_train)

        # Save pipeline components
        artifacts_path = "artifacts"
        os.makedirs(artifacts_path, exist_ok=True)

        # Save full pipeline
        joblib.dump(pipeline, os.path.join(artifacts_path, "pipeline.pkl"))
        from src.utils import save_object

# Saving the preprocessor
        save_object(os.path.join(artifacts_path, "preprocessor.pkl"), pipeline.named_steps['preprocessor'])

# Saving the model
        save_object(os.path.join(artifacts_path, "model.pkl"), pipeline.named_steps['model'])


        print("Training completed and artifacts saved!")

    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    train_model()
