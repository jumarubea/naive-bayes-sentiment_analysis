# Naive Bayes Sentiment Analysis

This repository implements a **Naive Bayes** classifier for **sentiment analysis**. The goal is to classify text data (tweets) into categories such as **positive** or **negative** sentiment. The project is built using Python and Jupyter Notebooks.

## Table of Contents
- [Project Overview](#project-overview)
- [Files and Directories](#files-and-directories)
- [Installation](#installation)
- [Usage](#usage)

## Project Overview

Sentiment analysis is a Natural Language Processing (NLP) task that involves determining the sentiment expressed in a piece of text. This project implements the **Naive Bayes** algorithm, which is a probabilistic model used for classification tasks, such as sentiment analysis.

The project uses the **Naive Bayes Rule** to classify text data into two categories: **positive** and **negative** sentiment.

## Files and Directories

Here is an overview of the important files and directories in this repository:

- **`main.ipynb`**: The main Jupyter Notebook where the data preprocessing, model training, and evaluation are done. It contains the core logic of the sentiment analysis.
  
- **`utils/`**: A directory containing utility functions used for preprocessing the data, such as text cleaning, tokenization, and vectorization.

- **`model/`**: Contains the model implementation, including training the Naive Bayes classifier and making predictions on test data.

- **`requirements.txt`**: A text file that lists the Python dependencies required to run the project.

- **`README.md`**: This file, which provides an overview of the project, installation, and usage instructions.

## Installation

To run this project, you need to have Python 3.x installed on your system. You can create a virtual environment and install the dependencies using the `requirements.txt` file.

1. Clone the repository:
   ```bash
   git clone https://github.com/jumarubea/naive-bayes-sentiment_analysis.git
   ```

2. Navigate to the project directory:

```bash
cd naive-bayes-sentiment_analysis
```
3. Create a virtual environment (optional but recommended):
``` bash
python -m venv venv
```
4. Activate the virtual environment:

On Windows:
```bash
venv\Scripts\activate
```
On macOS/Linux:
```bash
source venv/bin/activate
```
Install the required dependencies:

```bash
pip install -r requirements.txt
```
## Usage
Open the `main.ipynb` file in a Jupyter Notebook environment.

Run the notebook cells step-by-step to:

- Load and preprocess the dataset.
- Train the Naive Bayes classifier on the dataset.
- Evaluate the model's performance.
- Make predictions on new text data.
