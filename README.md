# Amazon Product Recommendation System (Data Engineering Project)

A scalable product recommendation system built using Python, collaborative filtering (SVD), and Flask. This project demonstrates end-to-end data engineering and machine learning workflow, including data preprocessing, model training, and web deployment.

---

## Project Overview

This project builds a personalized recommendation system that suggests products to users based on their historical ratings. It uses collaborative filtering and Singular Value Decomposition (SVD) to identify patterns in user behavior and predict ratings for products that users have not yet interacted with.

The project also includes a Flask-based web application that allows users to input a User ID and receive top recommended products.

This project demonstrates real-world Data Engineering and Machine Learning concepts used in production systems.

---

## Key Features

- Personalized product recommendations
- Collaborative filtering using matrix factorization (SVD)
- Flask web application interface
- Data preprocessing and transformation
- Scalable and efficient recommendation logic
- End-to-end pipeline from data loading to prediction

---

## Technology Stack

### Programming Language
- Python

### Libraries and Frameworks
- Pandas
- NumPy
- SciPy
- Scikit-learn
- Flask

### Data Engineering and Machine Learning Concepts
- Recommendation Systems
- Collaborative Filtering
- Matrix Factorization (SVD)
- Data Processing
- Model Training and Prediction

---

## Project Structure

Final_data_Engineering_Project/

- app.py → Flask web application
- train.py → Recommendation system logic
- templates/
  - index.html → Web interface
- ratings_Electronic.csv → Dataset (add locally if not included)
- Screenshots/ → Project screenshots
- demo vedio.mp4 → Demo video
- Presentation_Vedio.mp4 → Presentation video
- README.md → Project documentation

---

## How It Works

Step 1: Load user-product ratings dataset

Step 2: Preprocess and clean the dataset

Step 3: Create user-item matrix

Step 4: Apply Singular Value Decomposition (SVD)

Step 5: Predict ratings for unseen products

Step 6: Generate Top-N product recommendations

Step 7: Display results using Flask web interface

---

## Installation and Setup

### Step 1: Clone the repository

git clone https://github.com/mummadiroshanreddy/Final_data_Engineering_Project.git

cd Final_data_Engineering_Project

---

### Step 2: Install dependencies

pip install flask pandas numpy scipy scikit-learn matplotlib seaborn

---

### Step 3: Add dataset

Place the file below in the project root directory:

ratings_Electronic.csv

---

## Run the Application

python app.py

Open browser:

http://127.0.0.1:5000

Enter User ID and number of recommendations.

---

## Example

Input:

User ID: 10

Recommendations: 5

Output:

Top 5 recommended products based on predicted ratings.

---

## Skills Demonstrated

This project demonstrates skills in:

- Data Engineering
- Python Programming
- Recommendation Systems
- Data Processing and Transformation
- Machine Learning Implementation
- Flask Application Development
- End-to-End Project Development

---

## Future Improvements

- Deploy application to AWS, Azure, or GCP
- Docker containerization
- REST API development
- Improve recommendation accuracy
- Add real-time data processing
- Add user authentication

---

## Author

Roshan Reddy

Data Engineer | Python | Spark | AWS | Azure | GCP | Snowflake

LinkedIn:  
https://www.linkedin.com/in/roshan-reddy-mummadi-6b264a364/

GitHub:  
https://github.com/mummadiroshanreddy

---

## Project Status

Completed

This project is ready for demonstration and portfolio use.

---


