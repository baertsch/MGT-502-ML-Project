### MGT-502-ML-Project

__Group Tissot: Nadège Baertschi & Zélia Décaillet__

_May 2025_

<img src="https://github.com/baertsch/MGT-502-ML-Project/blob/main/app/image/Logo.png?raw=true" alt="description" width="300" />

# ReadingBuddy: Recommendation System for Book Rentals
> _"Let ReadingBuddy help you find your next read!"_

RUNNING AT: [click here to see app](https://mgt-502-ml-project-b9gvc6qsrd9qs39qmkbogq.streamlit.app/)

SEE OUR VIDEO ON: [click here to see video](https://youtu.be/bnvc_jrA6es)

---

## Table of Contents
**[Introduction](#introduction)**<br>
**[Exploratory Data Analysis](#eda)**<br>
**[Methodology](#methodology)**<br>
**[Our Models](#our-models)**<br>
**[First Model: Item-Item Collaborative Filtering](#first-model)**<br>
**[Second Model: User-User Collaborative Filtering](#second-model)**<br>
**[Third Model: Frequency-Based Collaborative Filterings](#third-model)**<br>
**[Fourth Model: Content-Based Filtering ](#fourth-model)**<br>
**[Fifth Model: Content-Based Filtering from Text Embeddings](#fifth-model)**<br>
**[Last Model: Hybrid Model](#last-model)**<br>
**[Model Comparison](#model-comparison)**<br>
**[Example Recommendations](#example-recommendations)**<br>
**[Required libraries](#required-libraries)


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
<img src="https://github.com/baertsch/MGT-502-ML-Project/blob/main/plots/top_10_authors.png?raw=true" alt="description" width="800" />

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
<img src="https://github.com/baertsch/MGT-502-ML-Project/blob/main/plots/top_10_genres2.png?raw=true" alt="description" width="800" />

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

Here, you can see a snippet of it. 

<img src="https://github.com/baertsch/MGT-502-ML-Project/blob/main/plots/heatmap1.png?raw=true" alt="description" width="800" />

For example, this matrix show that user 0 has interacted with many of the first books of the librabry.
---
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
---
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
---
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
---
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

---
### Last Model
#### Hybrid Recommender System

Lastly, we experimented with **hybridization**—the process of combining multiple of our previous models to leverage their individual strengths.

After extensive testing, we discovered that the **best-performing combination** came from integrating predictions from:

- **User-user collaborative filtering** using frequency-based matrix  
- **Item-item collaborative filtering** using frequency-based matrix  
- **TF-IDF content-based filtering**

We assigned a weight to each model's predicted matrix and used a loop through possible weight combinations between 0 and 1 (ensuring their sum equals 1):

```python
hybrid_pred = user_model * w1 + item_model * w2 + content_model * w3
```
---
## Model Comparison
| Model                                                           | Precision@10 | Recall@10 |
|-----------------------------------------------------------------|--------------|-----------|
| Model 1: Item-Item Collaborative Filtering                      | 0.0535       | 0.2651    |
| Model 2: User-User Collaborative Filtering                      | 0.0542       | 0.2921    |
| Model 3.1: Item-Item Collaborative Filtering (Frequency-Based)  | 0.0538       | 0.2647    |
| Model 3.2: User-User Collaborative Filtering (Frequency-Based)  | 0.0573       | 0.2892    |
| Model 4.1: Content-Based Filtering (data from Kaggle)           | 0.0374       | 0.2113    |
| Model 4.2: Content-Based Filtering (more complete data)         | 0.0411       | 0.2343    |
| Model 5: Text Embeddings (complete data)                        | 0.0372       | 0.2656    |
| Model 6: Hybrid Model  (Model 3.1 + Model 3.2 + Model 4.2)      | 0.0591       | 0.2955    |

While comparing our models using metrics like **Precision@10** and **Recall@10**, we observed that the scores remained relatively low across all models. This may initially suggest underperformance, but it's crucial to interpret these metrics in the context of our dataset and evaluation design.

On average, each user in the **test set** has only **2.61 interactions**. That means even in the best-case scenario—where the top-10 recommendations are perfect, only about **2 or 3 books can actually be found in the test set** to count as "relevant" recommendations.

This creates a natural ceiling for our precision and recall metrics:
- **Precision@10** is capped by the fact that only ~2 of the 10 recommended items *could possibly* be matched in the test set.
- **Recall@10** is likewise limited, because we're measuring against a very small number of actual test interactions.

In short: **we’re evaluating how well our models can recover ~2 known books from a list of 10**—not how useful or meaningful the rest of the list might be to the user in a real setting.

This means our **absolute scores may be low**, but the **relative differences between models remain meaningful**, and help us identify the strongest approaches.

### Which Model is the Best?
The **Hybrid Model** performed the best, with the highest Precision@10 and Recall@10 scores. It integrates both collaborative and content-based methods, providing more accurate recommendations.

## Example Recommendations
Let’s consider  real examples from our model.
### Example 1:
One standout example of model success is seen with **User 24**, whose reading history reveals a clear thematic focus on **sociocultural animation**, **social work**, and **qualitative research methods**.

**User 24’s Reading History Includes:**
- *Enjeux des territoires pour l'animation socioculturelle*
- *L'écologisation du travail social*
- *Les limites à la croissance*
- *Les méthodes qualitatives*
- *Guide de l’enquête de terrain*
- *L’animation professionnelle*
- *L’animation socioculturelle*
- *Intervention sociale et animation*
- *Les animateurs socioculturels*
- *Animation et animateurs : le sens de l'action*
- *Animation socioculturelle : pratiques multiples*
- *Conceptualiser l’animation socioculturelle*
- *L’animation socioculturelle : fondements, modèles, pratiques*

This user demonstrates highly specialized interest in **social sciences**, particularly within the **field of sociocultural animation**, a niche academic and professional area.

**Model Recommendations for User 24:**

The system recommended titles such as:

- *L'écologisation du travail social*
- *Animation socioculturelle professionnelle*
- *L’animation professionnelle*
- *Intervention sociale et animation*
- *Conceptualiser l’animation socioculturelle*
- *L’animation socioculturelle : fondements, modèles, pratiques*

Not only are these books **contextually relevant**, but many of them are either:
- **Books the user had already read**, indicating strong alignment with their known preferences
- **Closely related titles** authored by other professionals in the same field, published by the same academic presses (e.g. Éditions IES, L’Harmattan, La Découverte)

**Why This Is a Good Recommendation Case:**

- **Topical consistency**: The model correctly identifies books that belong to the exact field the user is engaged with.
- **No generic or irrelevant titles**: Unlike generic bestsellers, the recommendations are academic, niche, and specific.
- **Content alignment**: These books share overlapping keywords, themes, and publishers—attributes captured effectively by the content-based filtering component of the model.
- **Collaborative reinforcement**: The model likely picked up co-reading patterns from users with similar profiles.

**What This Shows:**

This example highlights the **strength of the hybrid model**:
- **Content-based filtering** helps when the user's interests are specialized and not shared by many.
- **Collaborative filtering** contributes by reinforcing connections to highly relevant items.
- **Frequency-aware predictions** help prioritize books that were not just interacted with, but returned to multiple times.

In real use cases like this one, **ReadingBuddy** shows its ability to **support deep exploration of a topic**, ideal for students, researchers, and professionals working within a specific field.

---
### Example 2:
User 5805 with strong engagement in media, communication, and sociopolitical themes had interacted with the following books:

- *Planète médias: géopolitique des réseaux* by Philippe Boulanger  
- *Médias publics et société numérique* by Patrick-Yves Badillo  
- *Les fabuleux pouvoirs de l'hypnose* by Betty Mamane  
- *L'explosion de la communication* by Philippe Breton  
- *Pratique du marketing* by François Courvoisier  
- *Le Petit Robert* (specific edition)

However, when looking at the top-10 recommendations for this user, several versions of **“Le Petit Robert”** appeared, often in different editions or formats. While this might seem reasonable at first—since the user did interact with one edition—the list was **overpopulated by variations of the same book**.

#### Why this happened

This is a clear case where our model reveals a **limitation stemming from data duplication**. In our dataset, multiple records exist for nearly identical items—like “Le Petit Robert”, that vary slightly in metadata (different authors listed, new ISBNs, minor changes in description or formatting).

Because our models (especially the content-based and item-item collaborative filtering ones) rely heavily on textual metadata and co-interaction patterns, they treat these as **distinct but highly similar items**. As a result:
- The **content-based similarity score** is very high between these versions
- The model treats each version as a unique but highly recommended item
- This leads to **redundant recommendations**, occupying valuable space in the top-k list

#### What it means

This behavior is not necessarily an error—it's an expected outcome of how the model computes similarity, but it reduces the **diversity and usefulness** of recommendations. The user is essentially being told to re-read the same book in slightly different wrappers.

---

### Conclusion

Unsurprisingly, our model performs best when the user exhibits **clear and specific preferences**. In the case of User 24, the consistent focus on a single domain, sociocultural animation—allowed the model to identify and recommend highly relevant books with precision and confidence.

On the other hand, users with **diverse or scattered reading interests** may receive recommendations biased toward the **most popular category** they’ve touched. This is particularly evident when one topic dominates the interaction matrix or metadata is duplicated across many editions of a popular book. In such cases, the model tends to **over-recommend items from dominant clusters**, potentially crowding out equally relevant but less represented subjects.

This reinforces the idea that **content-based and collaborative filtering approaches excel when user intent is focused**, and highlights an opportunity for future improvements in **diversity-aware recommendation strategies**.

---

## Required libraries
- pandas
- numpy
- sklearn
- seaborn
- matplotlib
- os



