import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Smart Recommender", page_icon="🛍️")

SIM_PATH = '../output/embeddings/task2_similarity.csv' 
SENT_PATH = '../output/recommendations.csv'

@st.cache_data
def load_all_data():
    try:
        if not os.path.exists(SIM_PATH) or not os.path.exists(SENT_PATH):
            st.error(f"Files not found! Check paths: {SIM_PATH}")
            return None, None
            
        df_sim = pd.read_csv(SIM_PATH)
        if 'product' in str(df_sim.columns[0]).lower():
            df_sim = pd.read_csv(SIM_PATH, skiprows=1, header=None)

        sentiment_df = pd.read_csv(SENT_PATH)
        return df_sim, sentiment_df
    except Exception as e:
        st.error(f" Load Error: {e}")
        return None, None

def get_numeric_col(df):
    for i in range(len(df.columns)-1, -1, -1):
        try:
            pd.to_numeric(df.iloc[:, i].head(10))
            return i
        except:
            continue
    return 2 

df_sim_long, sentiment_df = load_all_data()

if df_sim_long is not None:
    score_idx = get_numeric_col(df_sim_long)
    id1_idx = score_idx - 2
    id2_idx = score_idx - 1
    sent_col = 'avg_sentiment' if 'avg_sentiment' in sentiment_df.columns else 'sentiment_score'
    sent_map = dict(zip(sentiment_df['productId'].astype(str).str.strip().str.lower(), sentiment_df[sent_col]))

    def get_hybrid_recommendations(product_id, top_n=5):
        product_id = str(product_id).strip()
        c1 = df_sim_long.iloc[:, id1_idx].astype(str).str.strip()
        c2 = df_sim_long.iloc[:, id2_idx].astype(str).str.strip()
        matches = df_sim_long[(c1 == product_id) | (c2 == product_id)].copy()
        
        hybrid_dict = {} 
        
        for _, row in matches.iterrows():
            p_a, p_b = str(row.iloc[id1_idx]).strip(), str(row.iloc[id2_idx]).strip()
            other_pid = p_b if p_a == product_id else p_a

            if other_pid == product_id or "product" in other_pid.lower(): 
                continue

            try:
                sim_value = float(row.iloc[score_idx])
                sentiment_val = sent_map.get(other_pid.lower(), 0)
                normalized_sentiment = (sentiment_val + 1) / 2
                final_score = (sim_value * 0.7) + (normalized_sentiment * 0.3)
                if other_pid not in hybrid_dict or final_score > hybrid_dict[other_pid]:
                    hybrid_dict[other_pid] = final_score
            except:
                continue
        sorted_recs = sorted(hybrid_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:top_n]
    all_ids = pd.concat([df_sim_long.iloc[:, id1_idx], df_sim_long.iloc[:, id2_idx]]).astype(str)
    real_ids_list = sorted(all_ids[~all_ids.str.lower().str.contains('product|review|unnamed|nan')].unique())

st.title("E-commerce Smart Recommender 🛍️")
st.markdown("---")

if df_sim_long is not None:
    selected_product = st.selectbox(" Search or select a product you liked:", real_ids_list)

    if st.button(" Recommend Me!"):
        with st.spinner('Analyzing patterns...'):
            recommendations = get_hybrid_recommendations(selected_product)
        
        if recommendations:
            st.success(f"Found {len(recommendations)} great matches!")
            st.subheader("Top products for you:")
            for i, (pid, score) in enumerate(recommendations, 1):
                st.info(f"**#{i}** | Product ID: `{pid}`  \n**Match Score: {score:.2%}%**")
        else:
            st.warning("No similar products found for this item.")
else:
    st.error("Please check your data paths. Files could not be loaded.")