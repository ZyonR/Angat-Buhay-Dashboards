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

def return_filtered_data(df_wide,df_long,school,set_,class_):
    filtered_df_wide = df_wide.copy()
    filtered_df_long = df_long.copy()

    if school!="All Schools":
        filtered_df_wide = filtered_df_wide[filtered_df_wide["School"]==school]
        filtered_df_long = filtered_df_long[filtered_df_long["School"]==school]
    if set_!="All Sets":
        filtered_df_wide = filtered_df_wide[filtered_df_wide["Set"]==set_]
        filtered_df_long = filtered_df_long[filtered_df_long["Set"]==set_]
    if class_!="All Classes":
        filtered_df_wide = filtered_df_wide[filtered_df_wide["Class"]==class_]
        filtered_df_long = filtered_df_long[filtered_df_long["Class"]==class_]
    
    return filtered_df_wide,filtered_df_long

def deployDash(data,data_wide):
    ab_pink = "#fb3b9a"
    ab_blue = "#54bfc4"
    ab_accent = "#A9AEC9"

    #Initialize the Data
    bktk_df = data_wide.drop(columns=["#"])
    bktk_df_long = data.drop(columns=["#"])

    #Filter the Data
    col_a1,col_a2 = st.columns([1,2])
    with col_a1:
        st.write("## Choose Filtering Options")
        # For school filtering options
        filter_school = ["All Schools"]+bktk_df_long["School"].unique().tolist()
        choosen_school = st.selectbox(
            f"What School in {bktk_df_long['Area'].unique()[0]}?",
            options=filter_school,
            index=0,
            placeholder="Select School ...",
        )
        # For filtering per set of said school
        filter_set = ["All Sets"]+bktk_df_long[bktk_df_long["School"]==choosen_school]["Set"].unique().tolist()
        choosen_set = st.selectbox(
            f"At what set in {choosen_school}?",
            options=filter_set,
            index=0,
            placeholder=f"Select a set in {choosen_school} ...",
        )
        # For filtering per class of said set
        filter_class = ["All Classes"]+bktk_df_long[(bktk_df_long["School"]==choosen_school)&(bktk_df_long["Set"]==choosen_set)]["Class"].unique().tolist()
        choosen_class = st.selectbox(
            f"At what class in {choosen_school} of {choosen_set}?",
            options=filter_class,
            index=0,
            placeholder=f"Select a class in {choosen_school} of {choosen_set} ...",
        )
    # Create a more 
    df_wide,df_long = return_filtered_data(bktk_df,bktk_df_long,choosen_school,choosen_set,choosen_class)

    # Obtain Descriptive Statistics
    df_wide_num = df_wide.select_dtypes(include='number')
    df_w_agg=df_wide_num.describe()
    cv = df_wide_num.std() / df_wide_num.mean()
    df_w_agg.loc['cv'] = cv
    df_w_agg.loc['std'] = df_wide_num.std(ddof=0)
    df_w_agg_t = df_w_agg.T

    df_w_agg_t.columns = ['Count', 'Mean',"Standard Deviation","Minimum","25%","50%","75%","Maximum","Coefficent of Variation"]
    with col_a2:
        st.write(f"### Descriptive Statistics of {choosen_school} for {choosen_set} in {choosen_class}")
        st.write(df_w_agg_t)

    # Define each order of topics for visualization
    real_order = ["Alphabet Knowledge - Naming","Alphabet Knowledge - Sound","Decoding - Pantig","Decoding - Salita","Decoding - Parirala","Decoding - Pangungusap","Passage Reading","Comprehension"]
    
    # Group by topic and test type to obtain the mean score
    mean_scores = df_long.groupby(["Topics", "Test Type"])["Scores"].mean().reset_index()
    mean_scores["Topics"] = pd.Categorical(mean_scores["Topics"], categories=real_order, ordered=True)

    st.write(f"### Mean Statistics of {choosen_school} for {choosen_set} in {choosen_class}")
    col_b1,col_b2 = st.columns(2)
    with col_b1:
        # Facet plot on Mean for both Pre and Post Tests
        figure_mean_pre_v_post = px.bar(
            mean_scores,
            x="Topics",
            y="Scores",
            color="Test Type",
            facet_col="Test Type",
            title="Mean Values by Topic (Pretest vs Posttest)",
            color_discrete_map={"Pretest": ab_blue, "Posttest": ab_pink},
            text="Scores",
            category_orders={"Topics": real_order,"Test Type": ["Pretest","Posttest"]}
        )
        figure_mean_pre_v_post.update_traces(
            texttemplate='%{text:.2f}',
            textposition='outside'
        )
        figure_mean_pre_v_post.update_yaxes(range=[0, 100+15])

        st.plotly_chart(figure_mean_pre_v_post)
    with col_b2:
        # Plot the difference between the two tests
        mean_diff = mean_scores.pivot(index="Topics", columns="Test Type", values="Scores").reset_index()
        mean_diff["Mean Difference"] = mean_diff["Posttest"] - mean_diff["Pretest"]

        fig_mean_diff = px.bar(mean_diff,
                        x="Topics",
                        y="Mean Difference",
                        title="Mean Difference by Topic",
                        text="Mean Difference",
                        color_discrete_sequence=[ab_accent] 
                        )
        fig_mean_diff.update_traces(
            texttemplate='%{text:.2f}',
            textposition='outside'
        )
        fig_mean_diff.update_yaxes(range=[0,np.max(mean_diff["Mean Difference"])+10])
        st.plotly_chart(fig_mean_diff)
    
    # Create a coefficient by variation plot
    st.write(f"### Coefficient of Variation of {choosen_school} for {choosen_set} in {choosen_class}")
    std_scores = df_long.groupby(["Topics", "Test Type"])["Scores"].std(ddof=0).reset_index()
    cv_scores = pd.merge(mean_scores, std_scores, on=["Topics", "Test Type"], suffixes=("_mean", "_std"))
    cv_scores["Coefficient of Variation"] = cv_scores["Scores_std"] / cv_scores["Scores_mean"]
    fig_cv = px.bar(
        cv_scores,
        x="Topics",
        y="Coefficient of Variation",
        color="Test Type",
        facet_col="Test Type",
        title="Coefficient of Variation Values by Topic (Pretest vs Posttest)",
        color_discrete_map={"Pretest": ab_blue, "Posttest": ab_pink},
        text="Coefficient of Variation",
        category_orders={"Topics": real_order,"Test Type": ["Pretest","Posttest"]}
    )
    fig_cv.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside'
    )
    fig_cv.update_yaxes(range=[0, np.max(cv_scores["Coefficient of Variation"])+0.1])
    st.plotly_chart(fig_cv)

    col_c1, col_c2= st.columns([1,2])
    with col_c1:
        st.write(f"### Distributions of {choosen_school} for {choosen_set} in {choosen_class}")
    with col_c2:
        col_d1, col_d2= st.columns(2)
        with col_d1:
            chosen_topic = st.selectbox(
                "What topic do you want to explore?",
                list(df_long["Topics"].unique()),
                index=0,
                placeholder="Select topic ...",
            )
        with col_d2:
            chosen_bin_size = st.slider("Select a bin size", min_value=0, max_value=20, value=10)

    filtered_topic_df = df_long[df_long['Topics'] == chosen_topic]
    mean_scores_for_hist = filtered_topic_df.groupby("Test Type")["Scores"].mean().to_dict()

    fig_hist_gen = px.histogram(
        filtered_topic_df,
        x="Scores",
        nbins=chosen_bin_size,
        title=f"{chosen_topic} Score Distribution (Pretest vs Posttest)",
        facet_col="Test Type",
        color="Test Type",
        color_discrete_map={"Pretest": ab_blue, "Posttest": ab_pink}
    )
    for i, test_type in enumerate(["Pretest", "Posttest"], start=1):
        mean_value = mean_scores_for_hist.get(test_type)
        if mean_value:
            fig_hist_gen.add_vline(
                x=mean_value,
                line_width=1,
                line_dash="solid",
                line_color="black",
                col=i
            )
            fig_hist_gen.add_annotation(
                x=mean_value,
                y=1,
                text=f"Mean: {mean_value:.2f}",
                showarrow=False,
                xanchor="center",
                yanchor="top",
                yref="paper",
                xref=f"x{i}",
                font=dict(color="black"),
                bgcolor=ab_accent,
                borderpad=2  
            )

    st.plotly_chart(fig_hist_gen)

    col_e1, col_e2= st.columns(2)
    with col_e1:
        st.write(f"### Scatter Plot of Students' Raw Pretest and Posttest Scores from {choosen_school} for {choosen_set} in {choosen_class}")
    fig_scat = px.scatter(
        filtered_topic_df,
        x="Name of Child",
        y="Scores",
        title=f"{chosen_topic} Raw Scores of each Student (Pretest vs Posttest)",
        color="Test Type",
        color_discrete_map={"Pretest": ab_blue, "Posttest": ab_pink}
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
