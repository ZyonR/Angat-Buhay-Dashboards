import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import wilcoxon
import plotly.express as px

st.set_page_config(
        page_title="BKTK Angat Buhay",
        
    )
st.write("## ANGAT BUHAY Bayan Ko Titser Ko (BKTK)")

def deployDash(data):
    bktk_df = data.drop(columns=["No."])

    numeric_bktk_df = bktk_df.select_dtypes(include='number')

    bktk_agg=bktk_df.describe()
    cv = numeric_bktk_df.std() / numeric_bktk_df.mean()
    bktk_agg.loc['cv'] = cv
    bktk_agg.loc['std'] = numeric_bktk_df.std(ddof=0)

    bktk_df_topics = bktk_agg.columns
    bktk_agg_trans = bktk_agg.T

    bktk_agg_trans.columns = ['Count', 'Mean',"Standard Deviation","Minimum","25%","50%","75%","Maximum","Coefficent of Variation"]

    gradeCounts = bktk_df['Grade Level'].value_counts().reset_index()
    gradeCounts.columns = ['Grade Level', 'Count']

    fig = px.bar(
        gradeCounts,
        x='Grade Level',
        y='Count',
        color='Grade Level',
        title='Number of Students per Grade Level',
        labels={'Count': 'Number of Students'}
    )

    st.plotly_chart(fig)


    st.write(bktk_agg_trans)

    meanVal = list(bktk_agg_trans["Mean"])
    meanDf = pd.DataFrame(
        {
            "Topics":bktk_df_topics,
            "Mean Values": meanVal
        }
    )
    meanDf["Test Type"] = meanDf["Topics"].apply(
        lambda x: "Pretest" if "Pretest" in x else ("Posttest" if "Posttest" in x else None)
    )
    meanDf["Topics"] = meanDf["Topics"].str.replace(r"\s?\(?(Pretest|Posttest)\)?", "", regex=True)
    sorted_means = meanDf.sort_values(by="Mean Values",ascending=False)

    fig_mean = px.bar(sorted_means,x="Topics",y="Mean Values",color="Test Type",
        facet_col="Test Type",
        title="Mean Values by Topic (Pretest vs Posttest)")
    st.plotly_chart(fig_mean)


    cvVal = list(bktk_agg_trans["Coefficent of Variation"])
    cvDf = pd.DataFrame(
        {
            "Topics":bktk_df_topics,
            "Coefficent of Variation Values": cvVal
        }
    )
    cvDf["Test Type"] = cvDf["Topics"].apply(
        lambda x: "Pretest" if "Pretest" in x else ("Posttest" if "Posttest" in x else None)
    )
    cvDf["Topics"] = cvDf["Topics"].str.replace(r"\s?\(?(Pretest|Posttest)\)?", "", regex=True)
    sorted_cv = cvDf.sort_values(by="Coefficent of Variation Values",ascending=False)

    fig_cv = px.bar(sorted_cv,x="Topics",y="Coefficent of Variation Values",color="Test Type",
        facet_col="Test Type",
        title="Coefficent of Variation Values by Topic (Pretest vs Posttest)")
    st.plotly_chart(fig_cv)


    long_bktk_df = bktk_df.melt(id_vars=["Name of Child","Grade Level"], var_name="Topics", value_name="Score")
    long_bktk_df["Test Type"] = long_bktk_df["Topics"].apply(
        lambda x: "Pretest" if "Pretest" in x else ("Posttest" if "Posttest" in x else None)
    )
    long_bktk_df["Topics"] = long_bktk_df["Topics"].str.replace(r"\s?\(?(Pretest|Posttest)\)?", "", regex=True)

    col1, col2 = st.columns(2)

    with col1:
        topic = st.selectbox(
        "What topic do you want to explore?",
        list(long_bktk_df["Topics"].unique()),
        index=None,
        placeholder="Select topic ...",
        )
        st.write("You have chosen topic: ",topic)

    with col2:
        bin_num = st.slider("Select a bin size", min_value=0, max_value=20, value=10)
        st.write("With a bin size of: ",bin_num)

    filtered_bktk_df = long_bktk_df[long_bktk_df['Topics'] == topic]

    preTestScore = long_bktk_df[(long_bktk_df["Topics"] == topic) & (long_bktk_df["Test Type"] == "Pretest")]["Score"]
    postTestScore = long_bktk_df[(long_bktk_df["Topics"] == topic) & (long_bktk_df["Test Type"] == "Posttest")]["Score"]

    stat, p_value = wilcoxon(preTestScore, postTestScore)

    n = len(preTestScore)
    effectSize = stat / (n ** 0.5)

    fig_hist_gen = px.histogram(
        filtered_bktk_df,
        x="Score",
        nbins=bin_num,
        title=f"{topic} Score Distribution (Pretest vs Posttest)",
        facet_col="Test Type",
        color="Test Type"
    )

    st.write(f"Wilxon Signed Rank Test P-Value: ",p_value)
    st.write(f"Effect Size: ",round(effectSize,5))

    st.plotly_chart(fig_hist_gen)

data = st.file_uploader(label="Upload your .csv file")
if data:
    df = pd.read_csv(data)
    deployDash(df)