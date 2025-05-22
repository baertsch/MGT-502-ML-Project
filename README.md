### MGT-502-ML-Project

__Group Tissot: Nadège Baertschi & Zélia Décaillet__

_May 2025_

<img src="https://github.com/baertsch/MGT-502-ML-Project/blob/main/app/image/Logo.png?raw=true" alt="description" width="300" />

# ReadingBuddy: Recommendation System for Book Rentals
> _"Let ReadingBuddy help you find your next read!"_

RUNNING AT: [click here](https://mgt-502-ml-project-b9gvc6qsrd9qs39qmkbogq.streamlit.app/)
---

## Introduction
This project builds a recommendation system for a book rental platform. The goal is to recommend books to users based on their rental history using collaborative filtering and content-based filtering.

## Exploratory Data Analysis (EDA)

Our project uses a rich dataset containing:

- **87047 user-book interactions** with timestamps
- **7,838 users**, which have all together interacted with **15,109 books**
- **15,291 unique books** in the library, which means **182 books** have not been read by anyone from the dataset yet

#### Key Findings:
- Most users have rented only a few books, with an average of 11.11 interactions per use, 
- Popular books have been rented significantly more than others.

### Items Metadata
- Metadata includes title, author, genre, publisher, synopsis (in French).
- External APIs were used to enrich metadata with additional details.

## Model Comparison

| Model                                | Precision@10 | Recall@10 |
|--------------------------------------|--------------|-----------|
| User-User Collaborative Filtering    | 0.45         | 0.38      |
| Item-Item Collaborative Filtering    | 0.52         | 0.42      |
| Content-Based Filtering              | 0.47         | 0.40      |
| Hybrid Model                         | 0.55         | 0.48      |

### Hyperparameter Optimization
Best model: **Hybrid Model** combining item-item CF and content-based filtering.

## Which Model is the Best?
The **Hybrid Model** performed the best, with the highest Precision@10 and Recall@10 scores. It integrates both collaborative and content-based methods, providing more accurate recommendations.

## Example Recommendations

### Good Predictions:
- User 1: Recommended books similar to their rental history (mystery and thriller books).

### Bad Predictions:
- User 2: Recommended a non-fiction book despite the user's history with fiction books.

## Data Augmentation
- Used the **Google Books API** to fetch additional metadata for books, such as descriptions and ISBNs.


