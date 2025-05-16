import streamlit as st
import pandas as pd
from scipy.stats import wilcoxon
import plotly.express as px
import plotly.graph_objects as go
import pingouin as pg
from PIL import Image
import requests
from io import BytesIO
import numpy as np

st.set_page_config(
        page_title="BKTK Angat Buhay",
        layout="wide"
    )

def deployDash(data,data_wide):
    bktk_df = data_wide.drop(columns=["#"])
    bktk_df_long = data.drop(columns=["#"])

    numeric_bktk_df = bktk_df.select_dtypes(include='number')

    bktk_agg=bktk_df.describe()
    cv = numeric_bktk_df.std() / numeric_bktk_df.mean()
    bktk_agg.loc['cv'] = cv
    bktk_agg.loc['std'] = numeric_bktk_df.std(ddof=0)
    bktk_agg_trans = bktk_agg.T

    bktk_agg_trans.columns = ['Count', 'Mean',"Standard Deviation","Minimum","25%","50%","75%","Maximum","Coefficent of Variation"]
    st.write(bktk_agg_trans)

    col1,col2 = st.columns(2)
    with col1:
        sortHow = st.selectbox(
            "How do you want to sort your data?",
            ["None","Ascending","Descending"],
            index=0,
            placeholder="Select order ...",
        )
    with col2:
        school_options = ["All"]+bktk_df_long["School"].unique().tolist()
        filter_school = st.selectbox(
            "What school do you want to look at?",
            options=school_options,
            index=0,
            placeholder="Select School ...",
        )
    real_order = ["Alphabet Knowledge - Naming","Alphabet Knowledge - Sound","Decoding - Pantig","Decoding - Salita","Decoding - Parirala","Decoding - Pangungusap","Passage Reading","Comprehension"]
    if filter_school!="All":
        filtered_bktk_df_long = bktk_df_long[bktk_df_long["School"] == filter_school]
    else:
        filtered_bktk_df_long = bktk_df_long
    
    mean_scores = filtered_bktk_df_long.groupby(["Topics", "Test Type"])["Scores"].mean().reset_index()

    topic_means = mean_scores.groupby("Topics")["Scores"].mean()

    if sortHow == "Ascending":
        sorted_topics = topic_means.sort_values(ascending=True).index.tolist()
    elif sortHow == "Descending":
        sorted_topics = topic_means.sort_values(ascending=False).index.tolist()
    else:
        sorted_topics = real_order

    mean_scores["Topics"] = pd.Categorical(mean_scores["Topics"], categories=sorted_topics, ordered=True)

    fig_mean_pre = px.bar(
        mean_scores,
        x="Topics",
        y="Scores",
        color="Test Type",
        facet_col="Test Type",
        title="Mean Values by Topic (Pretest vs Posttest)",
        color_discrete_map={"Pretest": "lightblue", "Posttest": "pink"},
        text="Scores",
        category_orders={"Topics": sorted_topics,"Test Type": ["Pretest","Posttest"]}
    )

    fig_mean_pre.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside'
    )
    fig_mean_pre.update_yaxes(range=[0, 100+15])

    st.plotly_chart(fig_mean_pre)

    mean_diff = mean_scores.pivot(index="Topics", columns="Test Type", values="Scores").reset_index()
    mean_diff["Mean Difference"] = mean_diff["Posttest"] - mean_diff["Pretest"]

    mean_diff["Topics"] = pd.Categorical(mean_diff["Topics"], categories=real_order, ordered=True)
    if sortHow == "Ascending":
        mean_diff = mean_diff.sort_values(by="Mean Difference", ascending=True)
    elif sortHow == "Descending":
        mean_diff = mean_diff.sort_values(by="Mean Difference", ascending=False)
    
    fig_mean_diff = px.bar(mean_diff,
                      x="Topics",
                      y="Mean Difference",
                      title="Mean Difference by Topic",
                      text="Mean Difference"
                     )
    fig_mean_diff.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside'
    )
    fig_mean_diff.update_yaxes(range=[0,np.max(mean_diff["Mean Difference"])+10])
    st.plotly_chart(fig_mean_diff)

    std_scores = filtered_bktk_df_long.groupby(["Topics", "Test Type"])["Scores"].std(ddof=0).reset_index()

    topic_means = mean_scores.groupby("Topics")["Scores"].mean()

    if sortHow == "Ascending":
        sorted_topics = topic_means.sort_values(ascending=True).index.tolist()
    elif sortHow == "Descending":
        sorted_topics = topic_means.sort_values(ascending=False).index.tolist()
    else:
        sorted_topics = real_order

    mean_scores["Topics"] = pd.Categorical(mean_scores["Topics"], categories=sorted_topics, ordered=True)
    cv_scores = pd.merge(mean_scores, std_scores, on=["Topics", "Test Type"], suffixes=("_mean", "_std"))
    cv_scores["Coefficient of Variation"] = cv_scores["Scores_std"] / cv_scores["Scores_mean"]

    fig_cv = px.bar(
        cv_scores,
        x="Topics",
        y="Coefficient of Variation",
        color="Test Type",
        facet_col="Test Type",
        title="Coefficient of Variation Values by Topic (Pretest vs Posttest)",
        color_discrete_map={"Pretest": "lightblue", "Posttest": "pink"},
        text="Coefficient of Variation",
        category_orders={"Topics": sorted_topics,"Test Type": ["Pretest","Posttest"]}
    )

    fig_cv.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside'
    )
    fig_cv.update_yaxes(range=[0, np.max(cv_scores["Coefficient of Variation"])+0.1])

    st.plotly_chart(fig_cv)

    col1, col2,col3 = st.columns(3)

    with col1:
        topic = st.selectbox(
        "What topic do you want to explore?",
        list(bktk_df_long["Topics"].unique()),
        index=1,
        placeholder="Select topic ...",
        )
    with col2:
        school_options_hist = ["All"]+bktk_df_long["School"].unique().tolist()
        filter_school_hist = st.selectbox(
            "What school do you want to look at?",
            options=school_options_hist,
            index=0,
            placeholder="Select School ...",
            key="school_select_2",
        )
    with col3:
        bin_num = st.slider("Select a bin size", min_value=0, max_value=20, value=10)

    if filter_school_hist!="All":
        filtered_bktk_df = bktk_df_long[bktk_df_long["School"] == filter_school_hist]
    else:
        filtered_bktk_df = bktk_df_long

    filtered_bktk_df = filtered_bktk_df[bktk_df_long['Topics'] == topic]

    preTestScore = bktk_df_long[(bktk_df_long["Topics"] == topic) & (bktk_df_long["Test Type"] == "Pretest")]["Scores"]
    postTestScore = bktk_df_long[(bktk_df_long["Topics"] == topic) & (bktk_df_long["Test Type"] == "Posttest")]["Scores"]

    if topic:
        try:
            result = pg.wilcoxon(x=postTestScore,y=preTestScore,alternative='two-sided')
            st.write(result)
        except:
            ...

    fig_hist_gen = px.histogram(
        filtered_bktk_df,
        x="Scores",
        nbins=bin_num,
        title=f"{topic} Score Distribution (Pretest vs Posttest)",
        facet_col="Test Type",
        color="Test Type",
        color_discrete_map={"Pretest": "lightblue", "Posttest": "pink"}
    )
    st.plotly_chart(fig_hist_gen)

    st.title("Z-Scores")

    long_bktk_df_zscore = filtered_bktk_df.copy()
    for typeIter in list(long_bktk_df_zscore["Test Type"].unique()):
          for topicIter in list(long_bktk_df_zscore["Topics"].unique()):
                subset = long_bktk_df_zscore[
            (long_bktk_df_zscore["Topics"] == topicIter) & 
            (long_bktk_df_zscore["Test Type"] == typeIter)]
                subset_mean = subset["Scores"].mean()
                subset_std = subset["Scores"].std(ddof=0)

                long_bktk_df_zscore.loc[(long_bktk_df_zscore["Topics"] == topicIter) & (long_bktk_df_zscore["Test Type"] == typeIter), "Scores"] = (subset["Scores"] - subset_mean) / subset_std

    fig_scat = px.scatter(
        long_bktk_df_zscore[long_bktk_df_zscore["Topics"]==topic],
        x="Name of Child",
        y="Scores",
        title=f"{topic} Z-Score Distribution (Pretest vs Posttest)",
        color="Test Type",
        color_discrete_map={"Pretest": "lightblue", "Posttest": "pink"}
    )
    fig_scat.update_layout(
        height=600
    )
    st.plotly_chart(fig_scat)


url = "https://www.angatbuhay.ph/wp-content/uploads/2023/03/cropped-Angat-buhay-logo-1.png"
response = requests.get(url)
image = Image.open(BytesIO(response.content))
st.image(image, use_container_width =False,width=250)
col1,col2 = st.columns([1,2])
with col1:
    st.write("### Bayan Ko Titser Ko (BKTK)")
    st.write("##### Program Evaluation")
with col2:
    data = st.file_uploader(label="Upload your .csv file")
    if data:
        bktk_naga = pd.read_csv(data)
        bktk_naga_long = pd.melt(bktk_naga, id_vars=["#","Name of Child","Grade Level","Area","School","Set","Class"], var_name='Topics', value_name='Scores')
        bktk_naga_long["Test Type"] = bktk_naga_long["Topics"].apply(
                lambda x: "Pretest" if "Pretest" in x else ("Posttest" if "Posttest" in x else None)
            )
        bktk_naga_long["Topics"] = bktk_naga_long["Topics"].str.replace(r"\s?\(?(Pretest|Posttest)\)?", "", regex=True)
        bktk_naga_long["Scores"] = pd.to_numeric(bktk_naga_long["Scores"], errors='coerce')
        bktk_naga_long_cleaned = bktk_naga_long[bktk_naga_long["Scores"]<=100]
if data:
    deployDash(bktk_naga_long_cleaned,bktk_naga)

