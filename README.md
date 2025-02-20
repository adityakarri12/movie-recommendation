# Movie Recommendation System  

## Overview  
This is a **Movie Recommendation System** built using **Streamlit**, **Pandas**, and **Scikit-learn**. It provides personalized movie recommendations using two filtering techniques:  

- **Collaborative Filtering** (K-Nearest Neighbors)  
- **Content-Based Filtering** (Genre & Rating-based recommendations)  

## Features  
✅ Load and display movie and ratings datasets.  
✅ Extract movie release years from titles.  
✅ Provide recommendations based on user-selected movie.  
✅ Use collaborative filtering (KNN) for recommendations.  
✅ Use content-based filtering based on genre and rating.  
✅ Filter movies by genre and year before selection.  

## Installation  

### 1. Clone the Repository  
```bash
git clone https://github.com/adityakarri12/movie-recommendation.git
```

### 2. Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3. Run the Application  
```bash
streamlit run app.py
```

## Dataset  
The project uses two datasets:  
- **movies.csv** → Contains movie information like title, genre, and year.  
- **ratings.csv** → Contains user ratings for different movies.  

## How It Works  
1. **Collaborative Filtering**  
   - Uses KNN to find similar movies based on user ratings.  
   - Requires a minimum number of votes to filter noise.  

2. **Content-Based Filtering**  
   - Recommends movies with similar genres and ratings.  

## Technologies Used  
- **Python**  
- **Streamlit**  
- **Pandas**  
- **NumPy**  
- **Scikit-learn**  
- **SciPy**  

## Author  
👨‍💻 Developed by [Aditya Karri](https://www.linkedin.com/in/aditya-karri-7128a61b1)  

## License  
📝 This project is open-source under the MIT License.  

---  
Happy coding! 🚀🎮

