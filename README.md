### MGT-502-ML-Project

__Group Tissot: Nadège Baertschi & Zélia Décaillet__

_May 2025_

<img src="https://github.com/baertsch/MGT-502-ML-Project/blob/main/app/image/Logo.png?raw=true" alt="description" width="300" />

# ReadingBuddy: Recommendation System for Book Rentals
> _"Let ReadingBuddy help you find your next read!"_

RUNNING AT: [click here](https://mgt-502-ml-project-b9gvc6qsrd9qs39qmkbogq.streamlit.app/)
---

## Table of Contents
**[Introduction](#introduction)**<br>
**[Exploratory Data Analysis](#eda)**<br>
**[Methodology](#methodology)**<br>
--- 

## Introduction
> **When we read for leisure, everything is about discovery.**  
The joy of stumbling upon a story that grips us, a world we don’t want to leave, or a character that resonates, this is what makes reading magical. Yet, with so many books out there, finding that *perfect* next read can feel overwhelming. **ReadingBuddy** is here to make that discovery seamless and delightful.

> **When we read for education or research, discovery becomes essential.**  
The right book, the right reference, the right voice, these are not just helpful; they’re critical. But combing through massive catalogs, outdated systems, or relying solely on keyword search can be inefficient and frustrating. **ReadingBuddy** streamlines this process, helping users surface relevant sources efficiently and intelligently.

Whether you're a passionate reader, a student, or a researcher, **ReadingBuddy is your companion**, helping you uncover books that matter *to you*.

## Dataset Provided by Kaggle

The dataset powering ReadingBuddy consists of:

- `87047` user-book interactions with timestamps.
- `7,838` users, which have all together interacted with `15,109` books.
- `15,291` unique books in the library, which means `182` books have not been read by anyone from the dataset yet.
- Metadata for each book: `Title`, `Author`, `Publisher`, `Subjects`, `ISBN`

## EDA

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
This shows a typical long-tail distribution, where a few popular books are heavily engaged with, while most remain under the radar.

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
<img src="https://github.com/baertsch/MGT-502-ML-Project/blob/main/plots/top_10_authors.jpeg?raw=true" alt="description" width="800" />

### And lastly what about the top 10 genres with the most books?
<img src="https://github.com/baertsch/MGT-502-ML-Project/blob/main/plots/top_10_genres.png?raw=true" alt="description" width="800" />

This plot shows results that align with the list of the 10 most popular books. Here, you can see the unbalanced distribution of genres in our library. Indeed, comics and manga books are the most prevalent. 

### Any missing values in the dataset provided by Kaggle?
There are a quite a few NaN values in our book Metadata.
- Missing `Author`: `2,653`
- Missing `Subjects`: `2,223`
- Missing `ISBN`: `723`
- Missing `Publisher`: `25`

Therefore, we used API requests to fill in the as many missing values as possible, and add other info, such as the abstract and the url link of the book cover in order to respresent them in our app.

## Let's explore our extended data
### Any remaining missing values?
Thanks to the data completion and augmentation, we significantly narrowed down the number of missing values, particularly for the `Author` and the `Subjects` categories.
- Missing `Author`: `402`
- Missing `Subjects`: `475`
- Missing `ISBN`: `723`
- Missing `Publisher`: `15`

### Let's see if the most prevalent genres in the library are still the same, now that we know more about our library
<img src="https://github.com/baertsch/MGT-502-ML-Project/blob/main/plots/top_10_genres_2.png?raw=true" alt="description" width="800" />

Interestingly, we can see now that, albeit comics being still Top 1, we also have a lot of fiction books.

## Methodology

First, we built a few basic models, Item-Item Collaborative Filtering, User-User Collaborative Filtering and Content-Based Filtering, on the data provided by Kaggle. 

### Data Splitting
Rather than randomly splitting the dataset, we chose a **historical split** to simulate a real-world scenario where we train on past interactions and predict future ones.
- **80% oldest interactions** → used for **training**
- **20% most recent interactions** → used for **testing**

This method ensures that our model never "sees the future" and is trained only on data that would have been available at the time of recommendation.

### Building the Interaction Matrix
Next, we constructed a **binary user-item interaction matrix**:

- A **1** indicates that the user has interacted with the item
- A **0** means no interaction

This matrix is the core structure on which our collaborative filtering models operate.

To better understand the data structure, we visualized the matrix as a **heatmap**, revealing clusters of active users and popular items, as well as the sparsity of the dataset—typical of recommendation systems.

## Our Models
**Now that our ingredients are ready, let the cooking begin!

### First Model 
#### Item-Item Collaborative Filtering with Implicit Feedback
In this approach, we calculate the **cosine similarity** between items based on user interactions. The idea is that if a user interacted with item $i'$, and item $i$ is similar to $i'$, the user might also like $i$.

The predicted likelihood that user $u$ will interact with item $i$ is given by:

$$
P_u(i) = \frac{\sum\limits_{i' \in I} \text{sim}(i, i') \cdot R_u(i')}{\sum\limits_{i' \in I} \text{sim}(i, i')}
$$

**Explanation:**
- $P_u(i)$: predicted likelihood of user $u$ interacting with item $i$
- $\text{sim}(i, i')$: cosine similarity between item $i$ and item $i'$
- $R_u(i')$: 1 if user $u$ has interacted with item $i'$, 0 otherwise
- $I$: set of all items

This value $P_u(i)$ lies between 0 and 1, and represents how strong the recommendation is.

### Second Model
#### User-User Collaborative Filtering

Next, we explored the **user-user collaborative filtering** approach. Here, we compute the **cosine similarity** between users based on their item interactions.

The logic: if user $u'$ is similar to user $u$, and $u'$ interacted with item $i$, then $u$ might be interested in item $i$ too.

The predicted score is computed as:

$$
P_u(i) = \frac{\sum\limits_{u' \in U} \text{sim}(u, u') \cdot R_{u'}(i)}{\sum\limits_{u' \in U} \text{sim}(u, u')}
$$

**Explanation:**
- $P_u(i)$: likelihood of user $u$ interacting with item $i$
- $\text{sim}(u, u')$: cosine similarity between users $u$ and $u'$
- $R_{u'}(i)$: 1 if user $u'$ interacted with item $i$, 0 otherwise
- $U$: set of all users

### Third Model
#### From Binary to Frequency-Based Collaborative Filtering

To enrich the signal captured by our collaborative models, we replaced the binary interaction matrix (where 1 = interaction, 0 = no interaction) with a **frequency matrix**. In this matrix, each entry represents the **number of times** a user interacted with a given item.

This approach allows us to model **user engagement intensity** rather than just interaction presence. For example, a user who borrowed a book 10 times likely values it more than one they borrowed once.

We kept the same prediction function as in item-item collaborative filtering:

$$
P_u(i) = \frac{\sum\limits_{i' \in I} \text{sim}(i, i') \cdot R_u(i')}{\sum\limits_{i' \in I} \text{sim}(i, i')}
$$

Where:
- $R_u(i')$ is now the **number of interactions** between user $u$ and item $i'$
- All other terms remain the same

This simple adjustment gave our models a **more nuanced understanding** of preferences, particularly for users with rich interaction histories.

### Fourth Model
#### Content-Based Filtering Using Metadata

Next, we explored a **content-based approach**, which relies solely on item features (metadata) to drive recommendations.

We experimented with different combinations of available metadata:
- Title
- Subjects
- Author
- Publisher
- Synopsis

After empirical testing, we found that the best performance came from combining:
**`Title + Subjects + Author + Publisher`**

We concatenated the selected metadata fields into a single text string per book. Then we applied a **TF-IDF vectorizer**, which transforms the text into a numerical representation that captures the importance of terms across the corpus.

- Common words across many books are down-weighted
- Unique or distinguishing terms are given higher importance

This resulted in a high-dimensional **TF-IDF matrix**, where each row corresponds to a book and each column represents a term.n

We then computed **cosine similarity** between these TF-IDF vectors to measure how "alike" two books are in content. we used the **same prediction formula** as in item-item collaborative filtering:

$$
P_u(i) = \frac{\sum\limits_{i' \in I} \text{sim}(i, i') \cdot R_u(i')}{\sum\limits_{i' \in I} \text{sim}(i, i')}
$$

Here, the similarity $\text{sim}(i, i')$ is **content-based**, not interaction-based. This approach proved particularly valuable for:
- **Cold-start books** with no interaction history
- Capturing semantic relevance between books
---
### Fifth Model
#### Text Embeddings with Transformer Models

To go beyond the limitations of TF-IDF and better capture the **semantic meaning** of the metadata, we turned to **sentence transformers**.

We reused the same metadata structure as in the content-based model:
**`Title + Subjects + Author + Publisher`**

Each book's metadata was passed through a **pretrained SentenceTransformer model**, which encoded it into a dense vector embedding that captures rich semantic relationships between texts.

These **contextualized embeddings** outperform traditional bag-of-words methods by understanding nuance, such as:
- Synonyms (e.g., "thriller" vs. "suspense")
- Word order and contextual meaning
- Named entity similarities (e.g., similar author names)

After generating the book embeddings, we applied **cosine similarity** between books to measure content closeness—just like in the TF-IDF-based model.

We then predicted user preferences using the same formula as before:

$$
P_u(i) = \frac{\sum\limits_{i' \in I} \text{sim}(i, i') \cdot R_u(i')}{\sum\limits_{i' \in I} \text{sim}(i, i')}
$$


### Last Model
#### Hybrid Recommender System

Lastly, we experimented with **hybridization**—the process of combining multiple models to leverage their individual strengths.

After extensive testing, we discovered that the **best-performing combination** came from integrating predictions from:

- **User-user collaborative filtering** using frequency-based matrix  
- **Item-item collaborative filtering** using frequency-based matrix  
- **TF-IDF content-based filtering**

We assigned a weight to each model's predicted matrix and used a **grid search** approach to loop through possible weight combinations between 0 and 1 (ensuring their sum equals 1):

```python
hybrid_pred = user_model * w1 + item_model * w2 + content_model * w3
```
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


