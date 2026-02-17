# Amazon Product Recommendation System (Data Engineering Project)

A scalable product recommendation system built using Python, collaborative filtering (SVD), and Flask. This project demonstrates end-to-end data engineering and machine learning workflow, including data preprocessing, model training, and web deployment.

---

## Project Overview

This application generates personalized product recommendations for users based on historical ratings data. It uses matrix factorization (Singular Value Decomposition - SVD) to identify latent patterns between users and products and predict unseen ratings.

The project also includes a Flask web application where users can input their User ID and receive top recommended products.

---

## Key Features

- Personalized product recommendations using collaborative filtering
- Matrix factorization using SVD
- Interactive web interface using Flask
- Efficient data processing using Pandas and NumPy
- End-to-end implementation from data preprocessing to deployment
- Scalable design suitable for real-world recommendation systems

---

## Technology Stack

**Programming Language**
- Python

**Libraries and Frameworks**
- Pandas
- NumPy
- SciPy
- Scikit-learn
- Flask

**Concepts Used**
- Collaborative Filtering
- Matrix Factorization (SVD)
- Data Preprocessing
- Recommendation Systems
- Web Application Deployment

---

## Project Structure

Final_data_Engineering_Project/
│
├── app.py # Flask web application
├── train.py # Model training and recommendation logic
├── templates/
│ └── index.html # Web interface
├── ratings_Electronic.csv # Dataset (not included in repo if large)
├── Screenshots/ # Project screenshots
├── demo vedio.mp4 # Demo video
├── Presentation_Vedio.mp4 # Project presentation
└── README.md # Project documentation


---

## How It Works

1. Load and preprocess user-product ratings dataset
2. Create user-item interaction matrix
3. Apply Singular Value Decomposition (SVD)
4. Predict ratings for unseen products
5. Generate Top-N recommendations for the user
6. Display results using Flask web interface

---

## Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/mummadiroshanreddy/Final_data_Engineering_Project.git
cd Final_data_Engineering_Project

---

## How It Works

1. Load and preprocess user-product ratings dataset
2. Create user-item interaction matrix
3. Apply Singular Value Decomposition (SVD)
4. Predict ratings for unseen products
5. Generate Top-N recommendations for the user
6. Display results using Flask web interface

---

## Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/mummadiroshanreddy/Final_data_Engineering_Project.git
cd Final_data_Engineering_Project
Step 2: Install required dependencies
pip install flask pandas numpy scipy scikit-learn matplotlib seaborn

Step 3: Add dataset

Place the dataset file:

ratings_Electronic.csv


in the project root directory.

Run the Application
python app.py


Open your browser and go to:

http://127.0.0.1:5000


Enter a User ID and number of recommendations to generate results.

Example Use Case

Input:

User ID: 10

Recommendations: 5

Output:

Top 5 recommended products based on predicted ratings

Learning Outcomes

This project demonstrates practical knowledge of:

Data Engineering pipelines

Recommendation system development

Matrix factorization techniques

Python-based data processing

Flask application deployment

End-to-end project implementation

Future Improvements

Deploy using Docker and cloud platforms (AWS/GCP/Azure)

Improve model accuracy using advanced algorithms

Add real-time recommendation capability

Implement REST API

Add user authentication

Author

Roshan Reddy

Data Engineer | Python | Spark | AWS | Azure | GCP | Snowflake

LinkedIn:
https://www.linkedin.com/in/roshan-reddy-mummadi-6b264a364/

GitHub:
https://github.com/mummadiroshanreddy

Project Status

Completed – Ready for demonstration and further enhancement
