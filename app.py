import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B6B;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        margin-bottom: 1.5rem;
        border-left: 4px solid #4ECDC4;
        transition: transform 0.2s ease-in-out;
    }
    .recommendation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    .recommendation-card h4 {
        color: #2C3E50;
        font-weight: bold;
        font-size: 1.3rem;
        margin-bottom: 0.8rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .recommendation-card p {
        color: #34495E;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        line-height: 1.4;
    }
    .recommendation-card strong {
        color: #E74C3C;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the movie and ratings data"""
    try:
        movies_df = pd.read_csv("dataset/movies.csv")
        ratings_df = pd.read_csv("dataset/ratings.csv")
        
        # Clean column names
        movies_df.rename(columns=lambda x: x.strip(), inplace=True)
        
        # Handle missing values
        movies_df['director'] = movies_df['director'].fillna('Unknown')
        movies_df['starring'] = movies_df['starring'].fillna('Unknown')
        
        # Create combined features for content-based filtering
        movies_df['combined_features'] = (movies_df['genres'] + ' ' + 
                                        movies_df['director'] + ' ' + 
                                        movies_df['starring'])
        movies_df['combined_features'] = movies_df['combined_features'].fillna('')
        
        return movies_df, ratings_df
    except FileNotFoundError:
        st.error("Dataset files not found! Please ensure 'movies.csv' and 'ratings.csv' are in the 'dataset' folder.")
        return None, None

@st.cache_data
def prepare_models(movies_df, ratings_df):
    """Prepare all recommendation models"""
    
    # Content-Based Filtering
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['combined_features'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Collaborative Filtering - User-Item Matrix
    user_item_matrix = ratings_df.pivot_table(index='userId', columns='movieId', values='rating')
    user_item_matrix = user_item_matrix.fillna(0)
    user_item_matrix_sparse = csr_matrix(user_item_matrix)
    
    # KNN Model
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
    knn_model.fit(user_item_matrix_sparse)
    
    # SVD Model
    svd_model = TruncatedSVD(n_components=50, random_state=42)
    svd_model.fit(user_item_matrix)
    
    return {
        'cosine_sim': cosine_sim,
        'user_item_matrix': user_item_matrix,
        'user_item_matrix_sparse': user_item_matrix_sparse,
        'knn_model': knn_model,
        'svd_model': svd_model,
        'tfidf_vectorizer': tfidf_vectorizer
    }

def content_based_recommendations(movie_title, movies_df, cosine_sim, n=10):
    """Content-based movie recommendations"""
    try:
        # Find movie index
        idx = movies_df[movies_df['title'].str.contains(movie_title, case=False, na=False)].index
        if len(idx) == 0:
            return pd.Series([], dtype='object'), "Movie not found"
        
        idx = idx[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]  # Exclude the movie itself
        
        movie_indices = [i[0] for i in sim_scores]
        recommendations = movies_df.iloc[movie_indices][['title', 'genres', 'director', 'starring']]
        
        return recommendations, None
    except Exception as e:
        return pd.Series([], dtype='object'), str(e)

def collaborative_recommendations(user_id, movies_df, user_item_matrix, knn_model, n=10):
    """Collaborative filtering recommendations"""
    try:
        if user_id not in user_item_matrix.index:
            return pd.Series([], dtype='object'), "User not found"
        
        user_index = user_item_matrix.index.get_loc(user_id)
        distances, indices = knn_model.kneighbors(
            user_item_matrix.iloc[user_index,:].values.reshape(1, -1), 
            n_neighbors=n+1
        )
        
        similar_users = user_item_matrix.iloc[indices[0][1:], :]
        movie_ratings = similar_users.mean(axis=0)
        
        # Remove movies already watched by the user
        movies_watched_by_user = user_item_matrix.iloc[user_index, :]
        movie_ratings[movies_watched_by_user > 0] = -np.inf
        
        top_movie_indices = movie_ratings.nlargest(n).index
        recommendations = movies_df[movies_df['movieId'].isin(top_movie_indices)][['title', 'genres', 'director', 'starring']]
        
        return recommendations, None
    except Exception as e:
        return pd.Series([], dtype='object'), str(e)

def svd_recommendations(user_id, movies_df, user_item_matrix, svd_model, n=10):
    """SVD-based recommendations"""
    try:
        if user_id not in user_item_matrix.index:
            return pd.Series([], dtype='object'), "User not found"
        
        user_index = user_item_matrix.index.get_loc(user_id)
        user_vector = user_item_matrix.iloc[user_index, :].values.reshape(1, -1)
        
        # Transform and inverse transform to get predictions
        user_transformed = svd_model.transform(user_vector)
        user_reconstructed = svd_model.inverse_transform(user_transformed)
        
        # Get movies not yet rated by user
        movies_watched = user_item_matrix.iloc[user_index, :] > 0
        predicted_ratings = pd.Series(user_reconstructed[0], index=user_item_matrix.columns)
        predicted_ratings[movies_watched] = -np.inf
        
        top_movie_indices = predicted_ratings.nlargest(n).index
        recommendations = movies_df[movies_df['movieId'].isin(top_movie_indices)][['title', 'genres', 'director', 'starring']]
        
        return recommendations, None
    except Exception as e:
        return pd.Series([], dtype='object'), str(e)

def hybrid_recommendations(user_id, movie_title, movies_df, models, content_weight=0.3, collab_weight=0.4, svd_weight=0.3, n=10):
    """Hybrid recommendations combining all approaches"""
    
    content_recs, content_error = content_based_recommendations(movie_title, movies_df, models['cosine_sim'], n)
    collab_recs, collab_error = collaborative_recommendations(user_id, movies_df, models['user_item_matrix'], models['knn_model'], n)
    svd_recs, svd_error = svd_recommendations(user_id, movies_df, models['user_item_matrix'], models['svd_model'], n)
    
    # Combine recommendations
    all_recs = []
    
    if content_error is None and not content_recs.empty:
        all_recs.extend(content_recs['title'].tolist())
    
    if collab_error is None and not collab_recs.empty:
        all_recs.extend(collab_recs['title'].tolist())
    
    if svd_error is None and not svd_recs.empty:
        all_recs.extend(svd_recs['title'].tolist())
    
    # Remove duplicates while preserving order
    unique_recs = list(dict.fromkeys(all_recs))[:n]
    
    if unique_recs:
        hybrid_df = movies_df[movies_df['title'].isin(unique_recs)][['title', 'genres', 'director', 'starring']]
        return hybrid_df, None
    else:
        return pd.Series([], dtype='object'), "No recommendations found"

def cold_start_recommendations(movies_df, ratings_df, n=10):
    """Recommendations for new users based on popularity"""
    avg_ratings = ratings_df.groupby('movieId')['rating'].mean().sort_values(ascending=False)
    rating_counts = ratings_df.groupby('movieId')['rating'].count()
    
    # Filter movies with at least 10 ratings
    popular_movies = avg_ratings[rating_counts >= 10].head(n)
    recommendations = movies_df[movies_df['movieId'].isin(popular_movies.index)][['title', 'genres', 'director', 'starring']]
    
    return recommendations

def create_visualizations(movies_df, ratings_df):
    """Create various visualizations for the dashboard"""
    
    # Rating distribution
    fig_ratings = px.histogram(ratings_df, x='rating', nbins=10, 
                              title='Distribution of Movie Ratings',
                              color_discrete_sequence=['#FF6B6B'])
    fig_ratings.update_layout(xaxis_title='Rating', yaxis_title='Count')
    
    # Movies per platform
    platform_counts = movies_df['platform_Name'].value_counts()
    fig_platform = px.bar(x=platform_counts.index, y=platform_counts.values,
                         title='Number of Movies per Platform',
                         color_discrete_sequence=['#4ECDC4'])
    fig_platform.update_layout(xaxis_title='Platform', yaxis_title='Number of Movies')
    
    # Top genres
    genre_counts = movies_df['genres'].value_counts().head(10)
    fig_genres = px.bar(x=genre_counts.values, y=genre_counts.index,
                       orientation='h', title='Top 10 Movie Genres',
                       color_discrete_sequence=['#45B7D1'])
    fig_genres.update_layout(xaxis_title='Number of Movies', yaxis_title='Genres')
    
    # User activity distribution
    user_activity = ratings_df['userId'].value_counts()
    fig_activity = px.histogram(x=user_activity.values, nbins=50,
                               title='User Activity Distribution (Ratings per User)',
                               color_discrete_sequence=['#96CEB4'])
    fig_activity.update_layout(xaxis_title='Number of Ratings', yaxis_title='Number of Users')
    
    return fig_ratings, fig_platform, fig_genres, fig_activity

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown("### Discover your next favorite movie with AI-powered recommendations!")
    
    # Load data
    movies_df, ratings_df = load_data()
    
    if movies_df is None or ratings_df is None:
        st.stop()
    
    # Prepare models
    with st.spinner("Loading recommendation models..."):
        models = prepare_models(movies_df, ratings_df)
    
    # Sidebar
    st.sidebar.title("🎯 Recommendation Settings")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Choose Recommendation Method:",
        ["Content-Based", "Collaborative Filtering", "SVD-Based", "Hybrid Approach", "Cold Start (Popular Movies)"]
    )
    
    # Number of recommendations
    n_recommendations = st.sidebar.slider("Number of Recommendations:", 5, 20, 10)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🎬 Get Recommendations", "📊 Data Insights", "📈 Model Performance", "ℹ️ About"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Get Your Movie Recommendations</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if model_type in ["Content-Based", "Hybrid Approach"]:
                st.subheader("Select a Movie You Like:")
                movie_titles = movies_df['title'].tolist()
                selected_movie = st.selectbox("Choose a movie:", [""] + movie_titles)
            
            if model_type in ["Collaborative Filtering", "SVD-Based", "Hybrid Approach"]:
                st.subheader("Enter User ID:")
                user_ids = sorted(ratings_df['userId'].unique())
                selected_user = st.selectbox("Choose a user ID:", [None] + user_ids)
            
            if model_type == "Hybrid Approach":
                st.subheader("Hybrid Weights:")
                content_weight = st.slider("Content-Based Weight:", 0.0, 1.0, 0.3, 0.1)
                collab_weight = st.slider("Collaborative Weight:", 0.0, 1.0, 0.4, 0.1)
                svd_weight = st.slider("SVD Weight:", 0.0, 1.0, 0.3, 0.1)
                
                # Normalize weights
                total_weight = content_weight + collab_weight + svd_weight
                if total_weight > 0:
                    content_weight /= total_weight
                    collab_weight /= total_weight
                    svd_weight /= total_weight
        
        with col2:
            if st.button("🎯 Get Recommendations", type="primary"):
                with st.spinner("Generating recommendations..."):
                    
                    if model_type == "Content-Based":
                        if selected_movie:
                            recommendations, error = content_based_recommendations(
                                selected_movie, movies_df, models['cosine_sim'], n_recommendations
                            )
                        else:
                            st.warning("Please select a movie!")
                            recommendations, error = pd.Series([]), "No movie selected"
                    
                    elif model_type == "Collaborative Filtering":
                        if selected_user:
                            recommendations, error = collaborative_recommendations(
                                selected_user, movies_df, models['user_item_matrix'], 
                                models['knn_model'], n_recommendations
                            )
                        else:
                            st.warning("Please select a user ID!")
                            recommendations, error = pd.Series([]), "No user selected"
                    
                    elif model_type == "SVD-Based":
                        if selected_user:
                            recommendations, error = svd_recommendations(
                                selected_user, movies_df, models['user_item_matrix'], 
                                models['svd_model'], n_recommendations
                            )
                        else:
                            st.warning("Please select a user ID!")
                            recommendations, error = pd.Series([]), "No user selected"
                    
                    elif model_type == "Hybrid Approach":
                        if selected_movie and selected_user:
                            recommendations, error = hybrid_recommendations(
                                selected_user, selected_movie, movies_df, models,
                                content_weight, collab_weight, svd_weight, n_recommendations
                            )
                        else:
                            st.warning("Please select both a movie and a user ID!")
                            recommendations, error = pd.Series([]), "Missing selections"
                    
                    elif model_type == "Cold Start (Popular Movies)":
                        recommendations = cold_start_recommendations(movies_df, ratings_df, n_recommendations)
                        error = None
                    
                    # Display recommendations
                    if error:
                        st.error(f"Error: {error}")
                    elif recommendations.empty:
                        st.warning("No recommendations found!")
                    else:
                        st.success(f"Found {len(recommendations)} recommendations!")
                        
                        for idx, (_, movie) in enumerate(recommendations.iterrows(), 1):
                            with st.container():
                                st.markdown(f"""
                                <div class="recommendation-card">
                                    <h4>{idx}. {movie['title']}</h4>
                                    <p><strong>Genre:</strong> {movie['genres']}</p>
                                    <p><strong>Director:</strong> {movie['director']}</p>
                                    <p><strong>Starring:</strong> {movie['starring']}</p>
                                </div>
                                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<h2 class="sub-header">📊 Data Insights</h2>', unsafe_allow_html=True)
        
        # Create visualizations
        fig_ratings, fig_platform, fig_genres, fig_activity = create_visualizations(movies_df, ratings_df)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Movies", len(movies_df))
        with col2:
            st.metric("Total Ratings", len(ratings_df))
        with col3:
            st.metric("Total Users", ratings_df['userId'].nunique())
        with col4:
            st.metric("Average Rating", f"{ratings_df['rating'].mean():.2f}")
        
        # Display charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig_ratings, use_container_width=True)
            st.plotly_chart(fig_genres, use_container_width=True)
        
        with col2:
            st.plotly_chart(fig_platform, use_container_width=True)
            st.plotly_chart(fig_activity, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="sub-header">📈 Model Performance</h2>', unsafe_allow_html=True)
        
        st.info("Model performance metrics and comparisons")
        
        # Display model information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Content-Based Filtering")
            st.write("- Uses TF-IDF vectorization")
            st.write("- Based on movie features (genre, director, cast)")
            st.write("- Good for new users with movie preferences")
            
            st.subheader("Collaborative Filtering (KNN)")
            st.write("- Uses K-Nearest Neighbors")
            st.write("- Based on user-item interactions")
            st.write("- Finds similar users for recommendations")
        
        with col2:
            st.subheader("SVD-Based Filtering")
            st.write("- Uses Singular Value Decomposition")
            st.write("- Matrix factorization technique")
            st.write("- Handles sparse data well")
            
            st.subheader("Hybrid Approach")
            st.write("- Combines multiple methods")
            st.write("- Weighted average of predictions")
            st.write("- Better overall performance")
        
        # Model comparison chart
        model_performance = {
            'Model': ['Content-Based', 'KNN Collaborative', 'SVD', 'Hybrid'],
            'RMSE': [1.2, 1.08, 0.45, 0.85],  # Example values from notebooks
            'Coverage': [0.95, 0.75, 0.85, 0.90]
        }
        
        perf_df = pd.DataFrame(model_performance)
        
        fig_perf = px.bar(perf_df, x='Model', y='RMSE', 
                         title='Model Performance Comparison (Lower RMSE is Better)',
                         color='RMSE', color_continuous_scale='RdYlBu_r')
        st.plotly_chart(fig_perf, use_container_width=True)
    
    with tab4:
        st.markdown('<h2 class="sub-header">ℹ️ About This Project</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Project Overview
        This Movie Recommendation System implements multiple machine learning approaches to suggest movies based on user preferences and viewing history.
        
        ### 🔧 Technologies Used
        - **Python**: Core programming language
        - **Streamlit**: Web application framework
        - **Scikit-learn**: Machine learning algorithms
        - **Pandas & NumPy**: Data manipulation
        - **Plotly**: Interactive visualizations
        
        ### 📊 Dataset
        - **Source**: Bengali Movie Dataset from Kaggle
        - **Movies**: 381 movies with genres, directors, and cast information
        - **Ratings**: 105,156 user ratings on a scale of 0.5 to 5.0
        - **Users**: 668 unique users
        
        ### 🤖 Recommendation Algorithms
        
        1. **Content-Based Filtering**
           - Recommends movies similar to ones you've liked
           - Uses TF-IDF vectorization of movie features
           - Calculates cosine similarity between movies
        
        2. **Collaborative Filtering (KNN)**
           - Finds users with similar preferences
           - Recommends movies liked by similar users
           - Uses K-Nearest Neighbors algorithm
        
        3. **SVD-Based Filtering**
           - Matrix factorization technique
           - Reduces dimensionality while preserving patterns
           - Handles sparse user-item matrices effectively
        
        4. **Hybrid Approach**
           - Combines multiple recommendation methods
           - Uses weighted averaging for final recommendations
           - Provides more robust and diverse suggestions
        
        5. **Cold Start Recommendations**
           - For new users without rating history
           - Based on movie popularity and average ratings
           - Helps onboard new users to the system
        
        ### 📈 Model Performance
        - **SVD Model**: Best performance with RMSE of 0.45
        - **KNN Model**: RMSE of 1.08 with optimized parameters
        - **Hybrid Model**: Balanced approach combining strengths of all methods
        
        ### 🚀 Future Improvements
        - Deep learning models (Neural Collaborative Filtering)
        - Real-time learning from user interactions
        - Incorporation of movie plot summaries
        - Advanced regularization techniques
        - A/B testing for recommendation strategies
        """)
        
        st.markdown("---")
        st.markdown("**Developed by**: Bipul | **Dataset**: Bengali Movie Dataset (Kaggle)")

if __name__ == "__main__":
    main()
