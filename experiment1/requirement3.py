import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive environments (prevents tkinter errors)
import matplotlib.pyplot as plt
import seaborn as sns
import os

def set_chinese_font():
    """Attempts to set a common Chinese font."""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False  # Handle negative signs correctly
        print("Attempted to set Chinese font (e.g., SimHei, WenQuanYi Zen Hei, Microsoft YaHei).")
    except Exception as e:
        print(f"Could not set Chinese font: {e}. Plots will use default font.")

def create_visualizations(file_path, output_dir="experiment1"):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Attempt to set a font that supports Chinese characters
    set_chinese_font()

    # Student information text
    student_info = "Student ID: [YourStudentID], Name: [YourName]"

    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        print(f"Successfully read '{file_path}'")
        # Assuming standard column names for Iris dataset.
        # If your Excel uses Chinese column names, they need to be mapped or used directly.
        # Example: sepal_length_col = '萼片长度' (Sepal Length)
        #          sepal_width_col = '萼片宽度' (Sepal Width)
        #          species_col = '种类' (Species)
        # Based on previous run, the columns are: ['花瓣长度', '花瓣宽度', '类别']
        # The request was for Sepal length/width, but file provides Petal length/width.
        # Adapting scatter plot to use available Petal dimensions.
        x_axis_col = '花瓣长度' # Was 'Sepal length'
        y_axis_col = '花瓣宽度' # Was 'Sepal width'
        species_col = '类别'    # Was 'Species'

        print(f"Using columns: X='{x_axis_col}', Y='{y_axis_col}', Species='{species_col}' based on available data.")

        # Verify columns exist
        required_cols = [x_axis_col, y_axis_col, species_col]
        for col in required_cols:
            if col not in df.columns:
                print(f"Error: Critical Column '{col}' not found in the Excel file.")
                print(f"Available columns: {df.columns.tolist()}")
                return


    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except Exception as e:
        print(f"Error reading or processing Excel file: {e}")
        return

    # --- Matplotlib Scatter Plot ---
    try:
        plt.figure(figsize=(10, 6))
        # Assuming 'Species' column contains the categories for coloring
        unique_species = df[species_col].unique()
        # Using a common colormap; plt.cm.get_cmap is deprecated in favor of plt.colormaps[]
        try:
            colors = plt.colormaps.get_cmap('viridis')
        except AttributeError: # Fallback for older Matplotlib
            colors = plt.cm.get_cmap('viridis', len(unique_species))


        for i, species in enumerate(unique_species):
            species_data = df[df[species_col] == species]
            # For get_cmap that returns a ListedColormap, call it with i/len(unique_species)
            # For Colormap object from colormaps.get_cmap, it might need discrete colors differently
            # Simplest is to let scatter handle colors if possible, or use a list of colors
            color_val = colors(i / len(unique_species)) if callable(colors) and not isinstance(colors, list) else None


            plt.scatter(species_data[x_axis_col], species_data[y_axis_col],
                        color=color_val, label=species) # color=colors(i) might need adjustment based on cmap

        plt.title("Iris Petal Scatter Plot (Matplotlib)") # Title updated
        plt.xlabel(f"{x_axis_col} (cm)")
        plt.ylabel(f"{y_axis_col} (cm)")
        plt.legend(title=species_col)
        plt.grid(True)
        plt.text(0.99, 0.01, student_info, ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=8)
        scatter_plot_path = os.path.join(output_dir, "iris_matplotlib_scatter.png")
        plt.savefig(scatter_plot_path)
        plt.close() # Close the figure to free memory
        print(f"Matplotlib scatter plot saved to '{scatter_plot_path}'")
    except KeyError as e:
        print(f"Error creating Matplotlib plot (KeyError): Check column names. Missing: {e}")
        print(f"Available columns for Matplotlib plot: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error creating or saving Matplotlib scatter plot: {e}")


    # --- Seaborn Bar Chart ---
    try:
        plt.figure(figsize=(10, 6))
        # Count of samples per species using the correct species column name
        sns.countplot(x=species_col, data=df, palette="viridis", hue=species_col, legend=False)
        plt.title("Iris Species Sample Count (Seaborn)")
        plt.xlabel(species_col) # Use the correct column name for label
        plt.ylabel("Count of Samples")
        plt.grid(axis='y')
        plt.text(0.99, 0.01, student_info, ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=8)
        barchart_path = os.path.join(output_dir, "iris_seaborn_barchart.png")
        plt.savefig(barchart_path)
        plt.close() # Close the figure
        print(f"Seaborn bar chart saved to '{barchart_path}'")
    except KeyError as e:
        print(f"Error creating Seaborn plot (KeyError): Check column names. Missing: {e}")
        print(f"Available columns for Seaborn plot: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error creating or saving Seaborn bar chart: {e}")


if __name__ == "__main__":
    excel_file_path = "resources/Lab01-Iris.xlsx"
    create_visualizations(excel_file_path)
