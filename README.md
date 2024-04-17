# README

Welcome to our Python automation repository! This README provides essential information on how to use our provided batch files to run Python scripts either through a virtual environment or by dynamically installing required packages. Each approach is designed to suit different needs and setups.

## Prerequisites

Before you proceed, ensure the following requirements are met:

- **Python 3.11**: Assuming Python 3.11 is installed on your system. If not installed, you can download it from [python.org](https://www.python.org/downloads/).
- **Command Prompt (Windows)**: Ensure you have access to the Command Prompt as our batch files are tailored for Windows environments.

## Repository Structure

Here’s what you'll find in this repository:

- `hellowithenv.bat`: Sets up and activates a Python virtual environment, then runs a specified Python script.
- `hellonew.bat`: Dynamically installs required Python packages and then runs a Python script.


## Usage

### Approach 1: Using a Virtual Environment (`hellowithenv.bat`)

This batch file simplifies the process of setting up a Python virtual environment, activating it, and running a Python script within that environment.

#### Steps:

1. **Check for Virtual Environment**: The script checks if a virtual environment already exists in the specified directory.
2. **Environment Setup**: If no environment is found, it creates one using Python’s `venv` module.
3. **Activation**: Activates the virtual environment.
4. **Run Script**: Executes the Python script using the Python executable inside the virtual environment.

#### How to Run:

```bash
# Open Command Prompt
# Navigate to the directory containing hellowithenv.bat
hellowithenv.bat
or
just double click on batch file
```

### Approach 2: Installing Packages Dynamically (`hellonew.bat`)

This batch file is intended for those who prefer not to manage virtual environments and instead wish to install necessary Python packages dynamically.

#### Steps:

1. **Install Packages**: Uses `pip` to install all required packages from `requirements.txt`.
2. **Run Script**: Executes the Python script.

#### How to Run:

```bash
# Open Command Prompt
# Navigate to the directory containing hellonew.bat
hellonew.bat
or
just double click on batch file
```

## Best Practices

- **Update Tools**: Before running the scripts, it's recommended to update `pip` and `setuptools` to their latest versions:
  ```bash
  pip install --upgrade pip setuptools
  ```
- **Troubleshooting**: If you encounter issues related to package installations, verify that your Python setup is correct and that `pip` functions as expected.

