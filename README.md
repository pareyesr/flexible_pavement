# Flexible Pavement Design Tool

This application provides tools for flexible pavement design using AASHTO methodology.

## Setup Instructions

Follow these steps to set up and run the program:

1. **Clone the repository:**
    ```bash
    gh repo clone pareyesr/flexible_pavement
    cd flexible_pavement
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv env
    ```

3. **Activate the virtual environment:**
    - **Windows:**
      ```bash
      .\env\Scripts\activate
      ```
    - **macOS/Linux:**
      ```bash
      source env/bin/activate
      ```

4. **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

5. **Run the application:**
    ```bash
    python main.py
    ```

6. **When finished, deactivate the virtual environment:**
    ```bash
    deactivate
    ```

## Requirements

- Python 3.8 or higher
- All dependencies listed in `requirements.txt`
- The application includes GUI components using tkinter

## Features

- Structural Number (SN) calculation
- Material management and loading
- Pavement design solutions
- Traffic simulation and analysis
- Traditional design visualization
