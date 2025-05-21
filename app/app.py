import streamlit as st
import pandas as pd
import plotly.express as px

url = 'https://raw.githubusercontent.com/baertsch/MGT-502-ML-Project/refs/heads/main/hybrid_recommendation3.csv'
read = 'https://raw.githubusercontent.com/baertsch/MGT-502-ML-Project/refs/heads/main/kaggle_data/interactions_train.csv'
item = 'https://raw.githubusercontent.com/baertsch/MGT-502-ML-Project/refs/heads/main/kaggle_data/items_df.csv'

df = pd.read_csv(url)
read_df = pd.read_csv(read)
item_df = pd.read_csv(item)



if "page" not in st.session_state:
    st.session_state.page = "home"

# Navigation logic
def go_to_login():
    st.session_state.page = "login"

def go_to_recommend():
    st.session_state.page = "recommend"

if st.session_state.page == "home":
    st.title("Welcome to ReaddingBuddy")
    
    st.text("A book recommendation system that helps you find your next read!")

    col1, col2, col3 = st.columns([1,2,1])

    with col2: 
        st.image('./image/logo.png', width=400)
        st.text("Do you have an existing userId?")
        login_checked = st.checkbox("Yes, login here", value=False, key="login")
        st.checkbox("No, access recommendation based on book", value=False, key="recommend")
    with col3:
        st.markdown("<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        if st.button("Get Started", key="get_started"):
            go_to_recommend()
    if login_checked:
        go_to_login()

elif st.session_state.page == "login":
    col1, col2, col3 = st.columns([1,1,1])
    col2.image('./image/logo.png', width=200)
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
                        col.image('https://static.vecteezy.com/system/resources/previews/005/337/799/original/icon-image-not-found-free-vector.jpg', width=100)
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
                        col.image('https://static.vecteezy.com/system/resources/previews/005/337/799/original/icon-image-not-found-free-vector.jpg', width=100)
                    with col.expander(title):
                        st.write(f"**Author:** {author}")
                        st.write(f"**Publisher:** {publisher}")
                        st.write(f"**Synopsis:** {synopsis}") 
        else:
            st.write("No data found for the selected user ID.")

elif st.session_state.page == "recommend":
    col1, col2, col3 = st.columns([1,1,1])
    col2.image('./image/logo.png', width=200)
    st.header("Recommendation Page")
    if  col1.button("Back to Home"):
        st.session_state.page = "home"
    
    st.markdown("---")
    st.selectbox("Select a book", options=df['book_id'], key="book_id")