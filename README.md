### MGT-502-ML-Project

__Group Tissot: Nadège Baertschi & Zélia Décaillet__

_May 2025_

<img src="https://github.com/baertsch/MGT-502-ML-Project/blob/main/app/image/Logo.png?raw=true" alt="description" width="300" />

# ReadingBuddy: Recommendation System for Book Rentals
> _"Let ReadingBuddy help you find your next read!"_

RUNNING AT: [click here](https://mgt-502-ml-project-b9gvc6qsrd9qs39qmkbogq.streamlit.app/)
---

## Introduction
> **When we read for leisure, everything is about discovery.**  
The joy of stumbling upon a story that grips us, a world we don’t want to leave, or a character that resonates, this is what makes reading magical. Yet, with so many books out there, finding that *perfect* next read can feel overwhelming. **ReadingBuddy** is here to make that discovery seamless and delightful.

> **When we read for education or research, discovery becomes essential.**  
The right book, the right reference, the right voice—these are not just helpful; they’re critical. But combing through massive catalogs, outdated systems, or relying solely on keyword search can be inefficient and frustrating. **ReadingBuddy** streamlines this process, helping users surface relevant sources efficiently and intelligently.

Whether you're a passionate reader, a student, or a researcher, **ReadingBuddy is your companion**, helping you uncover books that matter *to you*.

## Dataset Provided by Kaggle

The dataset powering ReadingBuddy consists of:

- `87047` user-book interactions with timestamps.
- `7,838` users, which have all together interacted with `15,109` books.
- `15,291` unique books in the library, which means `182` books have not been read by anyone from the dataset yet.
- Metadata for each book: `Title`, `Author`, `Publisher`, `Subjects`, `ISBN`

## Exploratory Data Analysis (EDA)

Before diving into modeling, we explored our data deeply to understand its shape, quality, and potential.

### Let's see first how users interact with the library:
- **Average number of interactions per user**: `11.11`  
- **Median interactions per user**: `6`  
- **Maximum interactions (most active user)**: `385`  
- **Minimum**: `3`
Most users engage with a relatively small number of books, indicating a **sparse user-item interaction matrix**—a classic trait in recommendation systems that requires careful modeling.
This discrepancy between the average and the median can be explained by the skewed, unbalanced distribution of interactions. A small number of highly active users,who have interacted with hundreds of books, are pulling the average up significantly.
<img src="https://github.com/baertsch/MGT-502-ML-Project/blob/main/plots/interactions_per_user.png?raw=true" alt="description" width="800" />


### What about the number of interactions per item?
- **Average number of interactions per item**: `5.69`  
- **Median interactions per user**: `4`  
- **Maximum interactions (most active user)**: `380`  
- **Minimum**: `0`
- **Number of books never interacted with**: `182`
<img src="https://github.com/baertsch/MGT-502-ML-Project/blob/main/plots/interactions_per_item.png?raw=true" alt="description" width="800" />
This shows a typical __long-tail distribution__, where a few popular books are heavily engaged with, while most remain under the radar.

### What are the 10 most popular books?
We analyzed which books users engaged with the most:

| Rank | Title                       | Author              | Genre                         |
|------|-----------------------------|---------------------|-------------------------------|
| 1    | Le Petit Robert             | _N/A_               | French Dictionary             |
| 2    | Demon Slayer                | Gotōge, Koyoharu    | Manga                         |
| 3    | Vagabond                    | Inoué, Takehiko     | Manga                         |
| 4    | Spy x Family                | Endo, Tatsuya       | Manga / Fantasy               |
| 5    | L'Arabe du futur           | Sattouf, Riad       | Autobiographical BD           |
| 6    | The Promised Neverland      | Shirai, Kaiu        | Manga                         |
| 7    | Fullmetal Alchemist         | Arakawa, Hiromu     | Manga                         |
| 8    | Soins infirmiers            | Brunner, Lillian    | Nursing / Medical             |
| 9    | Pons Kompaktwörterbuch      | _N/A_               | French-German Dictionary      |
| 10   | Tokyo Revengers             | Wakui, Ken          | Manga                         |

More details, such as book cover and abstract, can be found on the home page of our app. Go check it out! You can also notice the popularity of Manga books!

### What about the top 10 Authors with the most books?
<img src="https://github.com/baertsch/MGT-502-ML-Project/blob/main/plots/top_10_auhtors.png?raw=true" alt="description" width="800" />

### And lastly what about the top 10 genres with the most books?
<img src="https://github.com/baertsch/MGT-502-ML-Project/blob/main/plots/top_10_genres.png?raw=true" alt="description" width="800" />
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


