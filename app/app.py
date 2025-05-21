import streamlit as st
import pandas as pd
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #e6c28a;
        color: black;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 18px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)


url = 'https://raw.githubusercontent.com/baertsch/MGT-502-ML-Project/refs/heads/main/hybrid_recommendation3.csv'
read = 'https://raw.githubusercontent.com/baertsch/MGT-502-ML-Project/refs/heads/main/kaggle_data/interactions_train.csv'
item = 'https://raw.githubusercontent.com/baertsch/MGT-502-ML-Project/refs/heads/main/kaggle_data/items_df.csv'
data = 'https://raw.githubusercontent.com/baertsch/MGT-502-ML-Project/refs/heads/main/item_to_item_recommendations.csv'
logo = 'https://github.com/baertsch/MGT-502-ML-Project/blob/main/app/image/Logo.png?raw=true'
book_cover = 'https://github.com/baertsch/MGT-502-ML-Project/blob/main/app/image/book_cover.png?raw=true'

df = pd.read_csv(url)
read_df = pd.read_csv(read)
item_df = pd.read_csv(item)
data = pd.read_csv(data)
pop_books = read_df.groupby('i').size().reset_index(name='count')
pop_books = pop_books.sort_values(by='count', ascending=False)
pop_books_idx = pop_books.iloc[:10]['i'].values.tolist()


if "page" not in st.session_state:
    st.session_state.page = "home"


if st.session_state.page == "home":
    
    st.title("Welcome to ReaddingBuddy!")
    
    st.text("A book recommendation system that helps you find your next read!")
    st.image(logo, width=400)
    st.markdown("---")
    st.subheader("The most popular books in our library:")
    n_cols = 3
    cols = st.columns(n_cols)
    for idx, book_id in enumerate(pop_books_idx):
        book_row = item_df[item_df['i'] == book_id][['Title', 'Author', 'Publisher', 'Synopsis', 'Image']]
        if not book_row.empty:
            img_url = book_row['Image'].values[0]
            title = book_row['Title'].values[0]
            author = book_row['Author'].values[0]
            publisher = book_row['Publisher'].values[0]
            synopsis = book_row['Synopsis'].values[0]
            col = cols[idx % n_cols]
            if isinstance(img_url, str) and img_url.strip() != "":
                col.image(img_url, width=100)
            else:
                col.image(book_cover, width=100)
            with col.expander(title):
                st.write(f"**Author:** {author}")
                st.write(f"**Publisher:** {publisher}")
                st.write(f"**Synopsis:** {synopsis}")
    st.markdown("---")

    col1, col2, col3 = st.columns([1,4,2])

    with col2: 
        st.text("Do you have an existing userId?")
        login_checked = st.checkbox("Yes, login here", value=False, key="login")
        recommend = st.checkbox("No, access recommendation based on book", value=False, key="recommend")
    
    #st.markdown("<br><br><br>", unsafe_allow_html=True)
    with col3:
        st.markdown("<br><br><br><br><br>", unsafe_allow_html=True) 
        if st.button("Get Started", key="get_started"):
            if login_checked:
                st.session_state.page = "login"
            elif st.session_state.recommend:
                st.session_state.page = "recommend"



elif st.session_state.page == "login":
    col1, col2, col3 = st.columns([1,1,1])
    col2.image(logo, width=200)
    st.header("User Recommender specifically targeted for you!")
    if col1.button("Back to Home"):
        st.session_state.page = "home"
    
    st.markdown("---")
    st.selectbox("Select your userId", options=df['user_id'], key="user_id")

    if st.button("Get Recommendations"):
        user_id = st.session_state.user_id
        user_data = read_df[read_df['u'] == user_id]
        book_data = user_data['i'].unique()
        rec_data = df[df['user_id'] == user_id]

        
        if not user_data.empty:
            

            rec_str = rec_data['recommendation'].values[0]
            rec_ids = [int(x) for x in rec_str.split(' ')]

            st.markdown("---")
            st.subheader("The books recommended for you:")
            n_cols = 3
            cols = st.columns(n_cols)
            for idx, book in enumerate(rec_ids):
                book_row = item_df[item_df['i'] == book][['Title', 'Author', 'Publisher', 'Synopsis', 'Image']]
                if not book_row.empty:
                    img_url = book_row['Image'].values[0]
                    title = book_row['Title'].values[0]
                    author = book_row['Author'].values[0]
                    publisher = book_row['Publisher'].values[0]
                    synopsis = book_row['Synopsis'].values[0]
                    col = cols[idx % n_cols]
                    if isinstance(img_url, str) and img_url.strip() != "":
                        col.image(img_url, width=100)
                    else:
                        col.image(book_cover, width=100)
                    with col.expander(title):
                        st.write(f"**Author:** {author}")
                        st.write(f"**Publisher:** {publisher}")
                        st.write(f"**Synopsis:** {synopsis}")
            
            st.markdown("---")
            st.subheader("The book you read:")
            n_cols = 3
            cols = st.columns(n_cols)
            for idx, book in enumerate(book_data):
                book_row = item_df[item_df['i'] == book][['Title', 'Author', 'Publisher', 'Synopsis', 'Image']]
                if not book_row.empty:
                    img_url = book_row['Image'].values[0]
                    title = book_row['Title'].values[0]
                    author = book_row['Author'].values[0]
                    publisher = book_row['Publisher'].values[0]
                    synopsis = book_row['Synopsis'].values[0]
                    col = cols[idx % n_cols]
                    if isinstance(img_url, str) and img_url.strip() != "":
                        col.image(img_url, width=100)
                    else:
                        col.image(book_cover, width=100)
                    with col.expander(title):
                        st.write(f"**Author:** {author}")
                        st.write(f"**Publisher:** {publisher}")
                        st.write(f"**Synopsis:** {synopsis}") 
        else:
            st.write("No data found for the selected user ID.")

elif st.session_state.page == "recommend":
    col1, col2, col3 = st.columns([1,1,1])
    col2.image(logo, width=200)
    st.header("Recommendation based on a book you like!")
    if  col1.button("Back to Home"):
        st.session_state.page = "home"
    
    st.markdown("---")
    st.selectbox("Select a book", options=item_df['Title'], key="book_title")
    if st.button("Get Recommendations"):
        book_title = st.session_state.book_title
        book_data = item_df[item_df['Title'] == book_title]
        if not book_data.empty:
            book_id = book_data['i'].values[0]
            rec_data = data[data['item_id'] == book_id]
            rec_str = rec_data['recommendation'].values[0]
            rec_ids = [int(x) for x in rec_str.split(' ')]
            
            st.markdown("---")
            st.subheader("The books recommended for you:")
            n_cols = 3
            cols = st.columns(n_cols)
            for idx, book in enumerate(rec_ids):
                book_row = item_df[item_df['i'] == book][['Title', 'Author', 'Publisher', 'Synopsis', 'Image']]
                if not book_row.empty:
                    img_url = book_row['Image'].values[0]
                    title = book_row['Title'].values[0]
                    author = book_row['Author'].values[0]
                    publisher = book_row['Publisher'].values[0]
                    synopsis = book_row['Synopsis'].values[0]
                    col = cols[idx % n_cols]
                    if isinstance(img_url, str) and img_url.strip() != "":
                        col.image(img_url, width=100)
                    else:
                        col.image(book_cover, width=100)
                    with col.expander(title):
                        st.write(f"**Author:** {author}")
                        st.write(f"**Publisher:** {publisher}")
                        st.write(f"**Synopsis:** {synopsis}") 
        else:
            st.write("No data found for the selected book.")
