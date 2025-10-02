# 🎬 Movie Recommendation System

A comprehensive movie recommendation system built with multiple machine learning approaches, featuring an interactive Streamlit web application for real-time movie recommendations.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.2-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

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
- **Interactive Web Interface**: Built with Streamlit for easy user interaction
- **Comprehensive Analysis**: Detailed exploratory data analysis and model evaluation
- **Real-time Recommendations**: Instant movie suggestions based on user input
- **Cold Start Handling**: Recommendations for new users without rating history
- **Performance Optimization**: Hyperparameter tuning and model comparison

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

## 📚 Dataset

**Source**: [Bengali Movie Dataset](https://www.kaggle.com/datasets/jocelyndumlao/bengali-movie-dataset) from Kaggle

### Dataset Statistics

- **Movies**: 381 Bengali movies
- **Ratings**: 105,156 user ratings
- **Users**: 668 unique users
- **Rating Scale**: 0.5 to 5.0 stars
- **Platforms**: Hoichoi (218 movies), Chorki (163 movies)

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
   - `Movie_Recommendation_System.ipynb`: Main implementation and analysis
   - `Movie_Recommendation_System__Model_Improvement.ipynb`: Advanced techniques and optimization

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
├── dataset/                          # Dataset files
│   ├── movies.csv                   # Movie information
│   └── ratings.csv                  # User ratings
│
├── notebooks/                       # Jupyter notebooks
│   ├── Movie_Recommendation_System.ipynb
│   └── Movie_Recommendation_System__Model_Improvement.ipynb
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                      # Project documentation
│
└── .gitignore                     # Git ignore file
```

## 📈 Model Performance

### Evaluation Metrics

| Algorithm | RMSE | Coverage | Strengths |
|-----------|------|----------|-----------|
| Content-Based | 1.20 | 95% | No cold start, interpretable |
| KNN Collaborative | 1.08 | 75% | Discovers user patterns |
| SVD | **0.45** | 85% | **Best performance**, handles sparsity |
| Hybrid | 0.85 | 90% | Balanced, robust |

### Key Findings

- **SVD performs best** with lowest RMSE (0.45)
- **Hybrid approach** provides good balance of performance and coverage
- **Content-based** offers highest coverage but moderate accuracy
- **KNN optimization** shows consistent performance across different K values

### Hyperparameter Tuning Results

- **Optimal K for KNN**: 5-10 neighbors
- **SVD Components**: 50 components provide best trade-off
- **Hybrid Weights**: 30% Content + 40% KNN + 30% SVD

## 📸 Screenshots

### Main Dashboard
![Dashboard](screenshots/dashboard.png)

### Recommendation Results
![Recommendations](screenshots/recommendations.png)

### Data Insights
![Insights](screenshots/insights.png)

*Note: Add actual screenshots to a `screenshots/` folder*

## 🔮 Future Improvements

### Short-term Enhancements

- [ ] **Deep Learning Models**: Implement Neural Collaborative Filtering
- [ ] **Real-time Learning**: Update models based on user interactions
- [ ] **A/B Testing**: Compare recommendation strategies
- [ ] **Movie Posters**: Add visual elements to recommendations

### Long-term Goals

- [ ] **Advanced NLP**: Incorporate movie plot summaries and reviews
- [ ] **Multi-modal Recommendations**: Include movie trailers and images
- [ ] **Explainable AI**: Provide reasons for recommendations
- [ ] **Social Features**: Friend-based recommendations
- [ ] **Mobile App**: React Native or Flutter implementation

### Technical Improvements

- [ ] **Caching**: Implement Redis for faster recommendations
- [ ] **Database**: Move from CSV to PostgreSQL/MongoDB
- [ ] **API**: Create RESTful API with FastAPI
- [ ] **Containerization**: Docker deployment
- [ ] **Cloud Deployment**: AWS/GCP/Azure hosting

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

```bash
# Clone and setup
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

**Happy Movie Watching! 🍿**
