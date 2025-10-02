# 🎬 Movie Recommendation System

A comprehensive movie recommendation system built with multiple machine learning approaches, featuring an interactive Streamlit web application for real-time movie recommendations.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.2-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌐 Live Demo

**Try the app live:** [https://movie-ai1.streamlit.app/](https://movie-ai1.streamlit.app/)

Experience the movie recommendation system in action with our deployed Streamlit application!

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Recommendation Algorithms](#recommendation-algorithms)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a sophisticated movie recommendation system using multiple machine learning approaches. The system can suggest movies based on user preferences, viewing history, and movie content features. It includes both Jupyter notebooks for experimentation and a production-ready Streamlit web application.

### Key Highlights

- **Multiple Recommendation Algorithms**: Content-based, Collaborative Filtering (KNN), SVD-based, and Hybrid approaches
- **Best-in-class Performance**: SVD model achieves RMSE of 0.454 (~58% improvement over KNN)
- **Interactive Web Interface**: Live Streamlit app deployed at [movie-ai1.streamlit.app](https://movie-ai1.streamlit.app/)
- **Comprehensive Analysis**: Two detailed Jupyter notebooks with EDA and model optimization
- **Real-time Recommendations**: Instant movie suggestions based on user input
- **Cold Start Handling**: Popularity-based recommendations for new users without rating history
- **Performance Optimization**: Extensive hyperparameter tuning and regularization techniques
- **Production-Ready**: Beautiful exception handling with Rich library integration

## ✨ Features

### 🤖 Recommendation Methods

1. **Content-Based Filtering**
   - Recommends movies similar to user's liked movies
   - Uses TF-IDF vectorization of movie features (genre, director, cast)
   - Calculates cosine similarity between movies

2. **Collaborative Filtering (KNN)**
   - Finds users with similar movie preferences
   - Uses K-Nearest Neighbors algorithm
   - Recommends movies liked by similar users

3. **SVD-Based Filtering**
   - Matrix factorization using Singular Value Decomposition
   - Handles sparse user-item matrices effectively
   - Best performing model with RMSE of 0.45

4. **Hybrid Approach**
   - Combines multiple recommendation methods
   - Weighted averaging of different algorithms
   - Provides more robust and diverse suggestions

5. **Cold Start Recommendations**
   - For new users without rating history
   - Based on movie popularity and average ratings

### 📊 Interactive Dashboard

- **Real-time Recommendations**: Get instant movie suggestions
- **Data Visualizations**: Interactive charts showing rating distributions, genre popularity, and user activity
- **Model Performance Metrics**: Compare different algorithms and their effectiveness
- **Customizable Parameters**: Adjust recommendation weights and number of suggestions

### 🔧 Technologies & Libraries

**Core Technologies:**
- **Python 3.8+**: Core programming language
- **Streamlit 1.28.1**: Web application framework for interactive UI
- **Scikit-learn 1.3.2**: Machine learning algorithms (KNN, SVD, TF-IDF)

**Data Processing:**
- **Pandas 2.1.3**: Data manipulation and analysis
- **NumPy 1.24.3**: Numerical operations and array processing
- **SciPy 1.11.4**: Sparse matrix operations for efficient computation

**Visualization:**
- **Plotly 5.17.0**: Interactive visualizations and charts
- **Matplotlib 3.8.2**: Static plotting and analysis
- **Seaborn 0.13.0**: Statistical data visualization

**Utilities:**
- **Rich 14.1.0+**: Beautiful exception logging and formatted terminal output
- **KaggleHub 0.2.5**: Automated dataset download and management

## 📚 Dataset

**Source**: [Bengali Movie Dataset](https://www.kaggle.com/datasets/jocelyndumlao/bengali-movie-dataset) from Kaggle

### Dataset Statistics

- **Movies**: 381 Bengali movies
- **Ratings**: 105,156 user ratings
- **Users**: 668 unique users
- **Rating Scale**: 0.5 to 5.0 stars (in 0.5 increments)
- **Platforms**: Hoichoi (218 movies), Chorki (163 movies)
- **Time Period**: Ratings from 1996 onwards
- **Average Rating**: ~3.7 stars
- **User Activity**: Users rated between 20 to 5,495 movies (mean: 157 ratings/user)

### Data Insights from EDA

- **Most Common Rating**: 4.0 stars (28,808 occurrences)
- **Rating Distribution**: Right-skewed with most ratings between 3.0-4.0
- **Top Genres**: Drama (96 movies), Thriller (43), Comedy (27), Horror (20)
- **Most Popular Movies**: Multiple movies with 276 ratings each
- **User Engagement**: Highly variable, with power users contributing significantly
- **Temporal Trends**: Average ratings show relative stability over time

### Movie Features

- **Title**: Movie name
- **Genres**: Movie categories (Drama, Thriller, Comedy, etc.)
- **Director**: Movie director
- **Starring**: Main cast members
- **Platform**: Streaming platform (Hoichoi/Chorki)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - The dataset will be automatically downloaded when you run the notebooks
   - Alternatively, download from [Kaggle](https://www.kaggle.com/datasets/jocelyndumlao/bengali-movie-dataset)
   - Place `movies.csv` and `ratings.csv` in the `dataset/` folder

## 💻 Usage

### Running the Streamlit App

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**
   - Navigate to `http://localhost:8501`
   - The app will automatically open in your default browser

### Using the Web Interface

1. **Choose Recommendation Method**
   - Select from Content-Based, Collaborative, SVD, Hybrid, or Cold Start
   - Adjust the number of recommendations (5-20)

2. **Provide Input**
   - For Content-Based: Select a movie you like
   - For Collaborative/SVD: Choose a user ID
   - For Hybrid: Provide both movie and user ID

3. **Get Recommendations**
   - Click "Get Recommendations" button
   - View personalized movie suggestions with details

4. **Explore Data Insights**
   - Check the "Data Insights" tab for visualizations
   - View model performance comparisons
   - Learn about the algorithms in the "About" section

### Running Jupyter Notebooks

1. **Start Jupyter**
   ```bash
   jupyter notebook
   ```

2. **Open notebooks**
   - **`Movie_Recommendation_System.ipynb`**: 
     - Start here for the complete implementation
     - Includes data loading, EDA, and basic models
     - Demonstrates content-based and collaborative filtering
     - Final RMSE: 1.083 for KNN model
   
   - **`Movie_Recommendation_System__Model_Improvement.ipynb`**: 
     - Advanced optimization techniques
     - SVD implementation with best performance (RMSE: 0.454)
     - Hyperparameter tuning and regularization
     - Cold start recommendations
     - Weighted hybrid approach

3. **Follow the project workflow**
   - Each notebook contains detailed markdown explanations
   - Code cells include comments and docstrings
   - Visualizations show data patterns and model performance
   - Results are displayed inline with analysis

## 🧠 Recommendation Algorithms

### 1. Content-Based Filtering

**How it works:**
- Analyzes movie features (genre, director, cast)
- Creates TF-IDF vectors for each movie
- Calculates cosine similarity between movies
- Recommends movies similar to user's preferences

**Advantages:**
- No cold start problem for items
- Transparent recommendations
- Works well for users with specific preferences

**Use Case:** "Users who liked 'Thriller' movies directed by 'XYZ' might also like..."

### 2. Collaborative Filtering (KNN)

**How it works:**
- Creates user-item interaction matrix
- Finds K nearest neighbors (similar users)
- Recommends movies liked by similar users
- Uses cosine similarity for user comparison

**Advantages:**
- Discovers hidden patterns in user behavior
- Can recommend diverse content
- Improves with more user data

**Use Case:** "Users similar to you also enjoyed these movies..."

### 3. SVD-Based Filtering

**How it works:**
- Performs matrix factorization on user-item matrix
- Reduces dimensionality while preserving patterns
- Predicts ratings for unrated movies
- Handles sparse data effectively

**Advantages:**
- Best performance (RMSE: 0.45)
- Handles sparsity well
- Scalable to large datasets

**Use Case:** Advanced pattern recognition in user preferences

### 4. Hybrid Approach

**How it works:**
- Combines predictions from multiple algorithms
- Uses weighted averaging (customizable weights)
- Leverages strengths of different methods
- Provides more robust recommendations

**Advantages:**
- Better overall performance
- Reduces individual algorithm weaknesses
- More diverse recommendations

**Default Weights:**
- Content-Based: 30%
- Collaborative (KNN): 40%
- SVD: 30%

## 📁 Project Structure

```
Movie_Recommendation_System/
│
├── dataset/                                           # Dataset files
│   ├── movies.csv                                    # Movie information (381 movies)
│   └── ratings.csv                                   # User ratings (105,156 ratings)
│
├── Movie_Recommendation_System.ipynb                 # Main implementation notebook
├── Movie_Recommendation_System__Model_Improvement.ipynb  # Advanced techniques & optimization
│
├── app.py                                           # Streamlit web application
├── requirements.txt                                 # Python dependencies
├── README.md                                       # Project documentation
│
└── .gitignore                                      # Git ignore file
```

### 📓 Notebook Details

**Movie_Recommendation_System.ipynb** - Core Implementation
- Complete project workflow from data loading to model evaluation
- Exploratory Data Analysis (EDA) with visualizations
- Implementation of content-based filtering using TF-IDF
- KNN-based collaborative filtering
- Initial hybrid approach combining both methods
- Model evaluation with RMSE metrics

**Movie_Recommendation_System__Model_Improvement.ipynb** - Advanced Optimization
- Hyperparameter tuning for KNN (testing k=5 to k=50)
- SVD implementation using TruncatedSVD (50 components)
- Weighted hybrid approach combining KNN and SVD
- Cold start solution for new users
- L2 regularization techniques for SVD
- Comprehensive model comparison and recommendations

## 📈 Model Performance

### Evaluation Metrics

| Algorithm | RMSE | Coverage | Strengths |
|-----------|------|----------|-----------|
| Content-Based | N/A | 95% | No cold start, interpretable |
| KNN Collaborative | 1.083 | 75% | Discovers user patterns |
| SVD | **0.454** | 85% | **Best performance**, handles sparsity |
| Hybrid (Weighted) | ~0.85 | 90% | Balanced, robust |

### Key Findings

- **SVD significantly outperforms other models** with lowest RMSE of **0.454**
- **KNN Collaborative Filtering** achieved consistent RMSE of **1.083** across different K values
- **Hyperparameter tuning** showed that KNN performance remained stable across k=5 to k=50
- **Hybrid approach** combines KNN and SVD predictions using weighted averaging
- **Cold start problem** addressed with popularity-based recommendations for new users

### Hyperparameter Tuning Results

- **Optimal K for KNN**: 5 neighbors (though performance was consistent from k=5 to k=50)
- **SVD Components**: 50 components provide optimal performance
- **Hybrid Weights**: 30% KNN + 40% SVD (adjustable based on use case)
- **Regularization**: L2 regularization applied to SVD to prevent overfitting

### Detailed Results from Notebooks

**Initial Implementation (Movie_Recommendation_System.ipynb):**
- KNN Collaborative Filtering RMSE: 1.0826
- Successfully implemented content-based, collaborative, and hybrid approaches
- Created comprehensive EDA with rating distributions and user activity analysis

**Model Improvement (Movie_Recommendation_System__Model_Improvement.ipynb):**
- SVD Model RMSE: 0.4535 (**~58% improvement over KNN**)
- Tested multiple regularization techniques
- Implemented cold start recommendations based on average movie ratings
- Explored weighted hybrid approach combining multiple algorithms

## 📸 Screenshots

### Main Dashboard
![Dashboard](screenshots/dashboard.png)

### Recommendation Results
![Recommendations](screenshots/recommendations.png)

### Data Insights
![Insights](screenshots/insights.png)

*Note: Add actual screenshots to a `screenshots/` folder*

## 🔮 Future Improvements

### Recommendations from Model Analysis

Based on the comprehensive analysis in the notebooks, here are the prioritized improvements:

**High Priority:**
- [ ] **Focus on SVD**: SVD model significantly outperforms others (RMSE: 0.454 vs 1.083 for KNN)
  - Further tune SVD parameters for even better performance
  - Experiment with different numbers of components (currently 50)
  
- [ ] **Refine Weighted Hybrid**: Improve the combination of KNN and SVD
  - Test different weight distributions
  - Consider giving more weight to SVD predictions
  - Implement dynamic weight adjustment based on user confidence

- [ ] **Feature Expansion**: Incorporate movie plot summaries
  - Enhance content-based filtering with textual features
  - Use advanced NLP techniques for plot analysis
  - Combine plot similarity with existing features

### Short-term Enhancements

- [ ] **Deep Learning Models**: Implement Neural Collaborative Filtering (NCF)
- [ ] **Advanced Regularization**: Implement more sophisticated regularization within SVD
- [ ] **Cold Start Optimization**: Enhance cold-start strategies for new users/movies
- [ ] **Real-time Learning**: Update models based on user interactions
- [ ] **A/B Testing**: Compare recommendation strategies systematically

### Long-term Goals

- [ ] **Multi-modal Recommendations**: Include movie trailers, posters, and reviews
- [ ] **Explainable AI**: Provide clear reasons for each recommendation
- [ ] **Social Features**: Friend-based and community recommendations
- [ ] **Mobile App**: React Native or Flutter implementation
- [ ] **Personalized UI**: Adaptive interface based on user preferences

### Technical Improvements

- [ ] **Caching**: Implement Redis for faster recommendations
- [ ] **Database**: Move from CSV to PostgreSQL/MongoDB
- [ ] **API**: Create RESTful API with FastAPI
- [ ] **Containerization**: Docker deployment (already deployed on Streamlit Cloud)
- [ ] **Model Monitoring**: Track model performance and drift over time
- [ ] **Automated Retraining**: Pipeline for periodic model updates

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [Bengali Movie Dataset](https://www.kaggle.com/datasets/jocelyndumlao/bengali-movie-dataset) by Jocelyn Dumlao
- **Inspiration**: Various recommendation system research papers and tutorials
- **Libraries**: Scikit-learn, Streamlit, Pandas, NumPy, and other open-source projects

## 📞 Contact

**Developer**: Bipul
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)
- **GitHub**: [Your GitHub Profile](https://github.com/yourusername)

---

⭐ **Star this repository if you found it helpful!**

## 🚀 Quick Start

### Option 1: Try the Live App (Fastest!)
Visit the deployed application: **[https://movie-ai1.streamlit.app/](https://movie-ai1.streamlit.app/)**

### Option 2: Run Locally

```bash
# Clone and setup
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### Option 3: Explore the Notebooks

```bash
# Install Jupyter
pip install jupyter

# Start Jupyter
jupyter notebook

# Open Movie_Recommendation_System.ipynb to get started
```

**Happy Movie Watching! 🍿**
