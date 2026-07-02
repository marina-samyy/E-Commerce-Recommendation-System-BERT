
import streamlit as st
import pandas as pd
import os


st.set_page_config(page_title="Amazon Recommendation System", layout="wide")


def find_file(relative_path):
    paths_to_check = [
        relative_path,
        os.path.join("..", relative_path),
        os.path.abspath(relative_path)
    ]
    for p in paths_to_check:
        if os.path.exists(p):
            return p
    return None

@st.cache_data
def load_data():
    base_path = "data/processed_transactions/association_rules.csv"
    rules_path = find_file(base_path)
    
    if rules_path:
        df = pd.read_csv(rules_path)
        df['antecedents'] = df['antecedents'].apply(lambda x: str(x).strip("frozenset({})' "))
        df['consequents'] = df['consequents'].apply(lambda x: str(x).strip("frozenset({})' "))
        return df
    return None

rules_df = load_data()

def get_recommendations(product_name, df, top_n=3):
    recs = df[df['antecedents'] == product_name]
    recs = recs.sort_values(by='lift', ascending=False)
    return recs['consequents'].unique()[:top_n]

st.title("🛒 Amazon Smart Recommender")
st.markdown("### Task 1: Data Mining & Link Analysis Insights")

if rules_df is not None:
    st.sidebar.header("Graph Statistics")
    st.sidebar.info(f"Total Rules Found: {len(rules_df)}")
    
    all_products = sorted(rules_df['antecedents'].unique())
    selected_product = st.selectbox("Select a product to see recommendations:", all_products)

    if st.button("Get Recommendations"):
        results = get_recommendations(selected_product, rules_df)
        
        if len(results) > 0:
            st.success(f"Customers who bought **{selected_product}** also bought:")
            cols = st.columns(len(results))
            for i, res in enumerate(results):
                with cols[i]:
                    st.info(f"**{res}**")
        else:
            st.warning("No specific recommendations found for this item.")

    st.divider()
    st.subheader("📊 Visualizing Data Insights")
    
    col1, col2 = st.columns(2)
    
    net_img = find_file("visualizations/networks/product_influence_network.png")
    bar_img = find_file("visualizations/charts/final_pagerank_bar_chart.png")

    with col1:
        st.write("#### Product Influence Network")
        if net_img:
            st.image(net_img, caption="PageRank Importance Network")
        else:
            st.warning("Network diagram image not found.")

    with col2:
        st.write("#### Top 10 Ranked Products")
        if bar_img:
            st.image(bar_img, caption="PageRank Scores Bar Chart")
        else:
            st.warning("Bar chart image not found.")

else:
    st.error(" association_rules.csv not found!")
    st.info(f"Current Directory: {os.getcwd()}")
    st.markdown("""
    **To fix this:**
    1. Open `fp_growth.ipynb`.
    2. Run the final cell that saves the CSV.
    3. Make sure the file exists in `data/processed_transactions/`.
    """)


