import pandas as pd
import numpy as np # numpy is a dependency of pandas, but good to import if using np features directly

def process_student_data(file_path):
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        print(f"Successfully read '{file_path}'")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    print("\nOriginal DataFrame head (for context, showing potential NaNs):")
    print(df.head())

    # Columns that are expected to be numeric
    expected_numeric_cols = ['年龄', '成绩']

    for col in expected_numeric_cols:
        if col in df.columns:
            # Convert column to numeric, coercing errors will turn non-numeric to NaNs
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                mean_val = df[col].mean() # Mean of the now numeric column
                df[col] = df[col].fillna(mean_val)
                print(f"Converted column '{col}' to numeric and filled NaNs with mean: {mean_val:.2f}")
            else:
                print(f"Column '{col}' is numeric and has no NaNs to fill.")
        else:
            print(f"Warning: Expected numeric column '{col}' not found in DataFrame.")


    # Display the first 5 rows
    print("\nFirst 5 rows of the processed DataFrame:")
    print(df.head(5))

    # Display the last 3 rows
    print("\nLast 3 rows of the processed DataFrame:")
    print(df.tail(3))

    # Calculate and print the average age and average score
    # Ensure columns '年龄' (Age) and '成绩' (Score) exist
    if '年龄' in df.columns and '成绩' in df.columns:
        average_age = df['年龄'].mean()
        average_score = df['成绩'].mean()
        print(f"\nAverage Age of all students: {average_age:.2f}")
        print(f"Average Score of all students: {average_score:.2f}")
    else:
        print("\nError: '年龄' or '成绩' column not found in the DataFrame.")
        if '年龄' not in df.columns:
            print("Missing column: 年龄 (Age)")
        if '成绩' not in df.columns:
            print("Missing column: 成绩 (Score)")
        print("Available columns:", df.columns.tolist())


if __name__ == "__main__":
    # Adjust the path if your script is not in the root or if resources is elsewhere
    excel_file_path = "resources/Lab01-Students.xlsx"
    process_student_data(excel_file_path)
