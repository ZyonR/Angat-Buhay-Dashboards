import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import requests
from io import BytesIO
import numpy as np

st.set_page_config(
        page_title="BKTK Angat Buhay",
        layout="wide")
def transform_data(data):
    df_long = pd.melt(data, id_vars=["Name of Child","Proxy Name","Grade Level","Area","School","Batch","Set"], var_name='Domains', value_name='Scores')
    df_long["Test Type"] = df_long["Domains"].apply(
            lambda x: "Pretest" if "Pretest" in x else ("Posttest" if "Posttest" in x else None)
        )
    df_long["Domains"] = df_long["Domains"].str.replace(r"\s?\(?(Pretest|Posttest)\)?", "", regex=True)
    df_long["Scores"] = pd.to_numeric(df_long["Scores"], errors='coerce')

    df_wide = df_long.pivot(index=["Name of Child","Proxy Name","Grade Level","Area","School","Batch","Set","Test Type"],
                        columns="Domains",
                        values="Scores").reset_index()
    return df_long,df_wide 
def custom_describe(data):
    desc = pd.DataFrame()
    desc["count"] = data.count()
    desc["mean"] = data.mean()
    desc["std (pop)"] = data.std(ddof=0)
    desc["min"] = data.min()
    desc["max"] = data.max()
    desc["cv"] = desc["std (pop)"] / desc["mean"]

    custom_order = ["Alphabet Knowledge - Naming","Alphabet Knowledge - Sound","Decoding - Pantig","Decoding - Salita","Decoding - Parirala","Decoding - Pangungusap","Passage Reading","Comprehension"]
    desc = desc.reindex(custom_order)
    return desc
def desc_the_data(data):
    df_pre = data[data["Test Type"]=="Pretest"]
    df_post = data[data["Test Type"]=="Posttest"]

    colc1,colc2 = st.columns(2)
    order = ["Alphabet Knowledge - Naming","Alphabet Knowledge - Sound","Decoding - Pantig","Decoding - Salita","Decoding - Parirala","Decoding - Pangungusap","Passage Reading","Comprehension"]
    with colc1:
        st.write("Pretest Scores of Sutudents")
        stats = custom_describe(df_pre.select_dtypes(include='number'))
        st.dataframe(stats)
        pre_mean = stats["mean"].tolist()
        pre_cv= stats["cv"].tolist()
    with colc2:
        st.write("Posttest Scores of Sutudents")
        stats = custom_describe(df_post.select_dtypes(include='number'))
        st.dataframe(stats)
        post_mean = stats["mean"].tolist()
        post_cv= stats["cv"].tolist()
    return pre_mean,post_mean,pre_cv,post_cv,order
def get_and_make_mean_df(pre,post,order):
    df_summary = pd.DataFrame({
        "Domain": order,
        "Pretest": pre,
        "Posttest": post
    })
    return df_summary
def get_and_make_cv_df(pre,post,order):
    df_summary = pd.DataFrame({
        "Domain": order,
        "Pretest": pre,
        "Posttest": post
    })
    return df_summary
def bar_viz(data,x_lab,y_lab,title,range=[0,1]):
    real_order = ["Alphabet Knowledge - Naming","Alphabet Knowledge - Sound","Decoding - Pantig","Decoding - Salita","Decoding - Parirala","Decoding - Pangungusap","Passage Reading","Comprehension"]
    ab_pink = "#fb3b9a"
    ab_blue = "#54bfc4"

    figure_mean_pre_v_post = px.bar(
            data,
            x=x_lab,
            y=y_lab,
            color="Test Type",
            title=title,
            color_discrete_map={"Pretest": ab_blue, "Posttest": ab_pink},
            text=y_lab,
            category_orders={"Domain": real_order,"Test Type": ["Pretest","Posttest"]}
        )
    figure_mean_pre_v_post.update_layout(barmode="group")
    figure_mean_pre_v_post.update_traces(
            texttemplate='%{text:.2f}',
            textposition='outside'
        )
    figure_mean_pre_v_post.update_yaxes(range=[range[0], range[1]])
    st.plotly_chart(figure_mean_pre_v_post)
def bar_viz_2(data,x_lab,y_lab,title,range=[-0.3,1]):
    real_order = ["Alphabet Knowledge - Naming","Alphabet Knowledge - Sound","Decoding - Pantig","Decoding - Salita","Decoding - Parirala","Decoding - Pangungusap","Passage Reading","Comprehension"]
    ab_accent = "#A9AEC9"

    figure_mean_pre_v_post = px.bar(
            data,
            x=x_lab,
            y=y_lab,
            title=title,
            text=y_lab,
            category_orders={"Domain": real_order},
            color_discrete_sequence=[ab_accent] 
        )
    figure_mean_pre_v_post.update_traces(
            texttemplate='%{text:.2f}',
            textposition='outside'
        )
    figure_mean_pre_v_post.update_yaxes(range=[range[0], range[1]])
    st.plotly_chart(figure_mean_pre_v_post)
def main_data_viz(df_long,df_wide):
    pre_mean,post_mean,pre_cv,post_cv,order = desc_the_data(df_wide)
    mean_df = get_and_make_mean_df(pre_mean,post_mean,order)
    mean_df_long = pd.melt(mean_df, id_vars=["Domain"], var_name='Test Type', value_name='Mean Scores')
    mean_df["Mean Difference"] = mean_df["Posttest"] - mean_df["Pretest"]
    diff_mean_min = mean_df["Mean Difference"].min()
    if diff_mean_min > 0:
        diff_mean_min = 0
    else:
        diff_mean_min = diff_mean_min-1

    cv_df = get_and_make_cv_df(pre_cv,post_cv,order)
    cv_df_long = pd.melt(cv_df, id_vars=["Domain"], var_name='Test Type', value_name='CV Scores')
    cv_df["CV Difference"] = cv_df["Posttest"] - cv_df["Pretest"]
    diff_cv_min = cv_df["CV Difference"].min()
    if diff_cv_min > 0:
        diff_cv_min = 0
    else:
        diff_cv_min = diff_cv_min-0.3
    
    cold1,cold2 = st.columns([2,1])
    with cold1:
        bar_viz(mean_df_long,"Domain","Mean Scores","Mean Scores for all Domains",[0,115])
    with cold2:
        bar_viz_2(mean_df,"Domain","Mean Difference","Mean Difference Scores for all Domains",[diff_mean_min,115])

    cole1,cole2 = st.columns([2,1])
    with cole1:
        bar_viz(cv_df_long,"Domain","CV Scores","CV Scores for all Domains")
    with cole2:
        bar_viz_2(cv_df,"Domain","CV Difference","CV Difference Scores for all Domains",[diff_cv_min,1])
def enable_filter(df_long, df_wide):
    st.write("## Choose Filtering Options")

    # 1. Area
    filter_area = ["All Areas"] + df_long["Area"].unique().tolist()
    choosen_area = st.selectbox(
        "What Area of the Philippines do you want to look at?",
        options=filter_area,
        index=0,
        placeholder="Select Area ..."
    )

    # 2. School
    if choosen_area != "All Areas":
        filter_school = ["All Schools"] + df_long[df_long["Area"] == choosen_area]["School"].unique().tolist()
    else:
        filter_school = ["All Schools"]
    choosen_school = st.selectbox(
        f"What School in {choosen_area}?",
        options=filter_school,
        index=0,
        placeholder="Select School ..."
    )

    # 3. Batch
    if choosen_area != "All Areas" and choosen_school != "All Schools":
        filter_batch = ["All Batch"] + df_long[
            (df_long["Area"] == choosen_area) &
            (df_long["School"] == choosen_school)
        ]["Batch"].unique().tolist()
    else:
        filter_batch = ["All Batch"]
    choosen_batch = st.selectbox(
        f"At what Batch in {choosen_school}?",
        options=filter_batch,
        index=0,
        placeholder=f"Select a Batch in {choosen_school} ..."
    )

    # 4. Set
    if choosen_area != "All Areas" and choosen_school != "All Schools" and choosen_batch != "All Batch":
        filter_set = ["All Set"] + df_long[
            (df_long["Area"] == choosen_area) &
            (df_long["School"] == choosen_school) &
            (df_long["Batch"] == choosen_batch)
        ]["Set"].unique().tolist()
    else:
        filter_set = ["All Set"]
    choosen_set = st.selectbox(
        f"At what Set in {choosen_school} of {choosen_batch}?",
        options=filter_set,
        index=0,
        placeholder=f"Select a Set in {choosen_school} of {choosen_batch} ..."
    )

    return choosen_area, choosen_school, choosen_batch, choosen_set
def population_std(x):
    return x.std(ddof=0)
def return_filtered_data(df_wide,df_long,area,school,set_,class_):
    filtered_df_wide = df_wide.copy()
    filtered_df_long = df_long.copy()
    
    if area!="All Areas":
        filtered_df_wide = filtered_df_wide[filtered_df_wide["Area"]==area]
        filtered_df_long = filtered_df_long[filtered_df_long["Area"]==area]
    if school!="All Schools":
        filtered_df_wide = filtered_df_wide[filtered_df_wide["School"]==school]
        filtered_df_long = filtered_df_long[filtered_df_long["School"]==school]
    if set_!="All Batch":
        filtered_df_wide = filtered_df_wide[filtered_df_wide["Batch"]==set_]
        filtered_df_long = filtered_df_long[filtered_df_long["Batch"]==set_]
    if class_!="All Set":
        filtered_df_wide = filtered_df_wide[filtered_df_wide["Set"]==class_]
        filtered_df_long = filtered_df_long[filtered_df_long["Set"]==class_]
    
    return filtered_df_wide,filtered_df_long
def create_histogtam(df_long,school,batch,set_):
    ab_pink = "#fb3b9a"
    ab_blue = "#54bfc4"
    ab_accent = "#A9AEC9"

    col_c1, col_c2= st.columns([1,2])
    with col_c1:
        st.write(f"### Distributions of {school} for {batch} in {set_}")
    with col_c2:
        col_d1, col_d2= st.columns(2)
        with col_d1:
            chosen_topic = st.selectbox(
                "What topic do you want to explore?",
                list(df_long["Domains"].unique()),
                index=0,
                placeholder="Select topic ...",
            )
        with col_d2:
            chosen_bin_size = st.slider("Select a bin size", min_value=0, max_value=20, value=10)

    filtered_topic_df = df_long[df_long['Domains'] == chosen_topic]
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
    return chosen_topic
def create_scatterplot(df_long,school,batch,set_,topic):
    ab_pink = "#fb3b9a"
    ab_blue = "#54bfc4"
    filtered_topic_df = df_long[df_long['Domains'] == topic]
    col_e1, col_e2= st.columns(2)
    with col_e1:
        st.write(f"### Scatter Plot of Students' Raw Pretest and Posttest Scores from {school} for {batch} in {set_}")
    fig_scat = px.scatter(
        filtered_topic_df,
        x="Name of Child",
        y="Scores",
        title=f"{topic} Raw Scores of each Student (Pretest vs Posttest)",
        color="Test Type",
        color_discrete_map={"Pretest": ab_blue, "Posttest": ab_pink}
    )
    fig_scat.update_layout(
        height=600
    )
    st.plotly_chart(fig_scat)
def site_dependent_dash(site,df_wide,df_long):
    ab_pink = "#fb3b9a"
    ab_blue = "#54bfc4"
    ab_accent = "#A9AEC9"

    st.title(f"Comparison Statistics for the Site of {site}")
    df_grouped = df_long.groupby(['School', 'Domains', 'Test Type'])["Scores"].agg(
        Mean = "mean",
        Std = population_std
    ).reset_index()
    df_grouped["CV"] = df_grouped["Std"]/df_grouped["Mean"]

    col_f1,col_f2 = st.columns(2)
    for topic in df_grouped["Domains"].unique().tolist():
        per_topic_df = df_grouped[df_grouped["Domains"] == topic]

        fig = px.bar(
            per_topic_df,
            x="School",
            y="Mean",
            color="Test Type",
            title=f"{topic} Mean Values per School (Pretest vs Posttest)",
            color_discrete_map={"Pretest": ab_blue, "Posttest": ab_pink},
            text="Mean",
            category_orders={"Test Type": ["Pretest","Posttest"]}
        )
        fig.update_traces(
            texttemplate='%{text:.2f}',
            textposition='outside'
        )
        fig.update_layout(barmode="group")
        fig.update_yaxes(range=[0, 100+15])
        with col_f1:
            st.plotly_chart(fig)
        fig = px.bar(
            per_topic_df,
            x="School",
            y="CV",
            color="Test Type",
            title=f"{topic} Coefficient of Variation Values per School (Pretest vs Posttest)",
            color_discrete_map={"Pretest": ab_blue, "Posttest": ab_pink},
            text="CV",
            category_orders={"Test Type": ["Pretest","Posttest"]}
        )
        fig.update_traces(
            texttemplate='%{text:.2f}',
            textposition='outside'
        )
        fig.update_layout(barmode="group")
        fig.update_yaxes(range=[0, 1.1])
        with col_f2:
            st.plotly_chart(fig)
def main_page():
    # Angat Buhay Logo
    url = "https://www.angatbuhay.ph/wp-content/uploads/2023/03/cropped-Angat-buhay-logo-1.png"
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))

    cola1,cola2 = st.columns([2,1])
    # Filtering Tab (Uploading and Filtering per Area and School)
    with cola2:
        data = st.file_uploader(label="Upload your .csv file")
    with cola1:
        # AB Logo and the Program Title
        colb1,colb2 = st.columns([1,2])
        with colb1:
            st.image(image, use_container_width =False,width=225)
        with colb2:
            st.title("BAYAN KO TITSER KO!")
            st.write("#### Program Monitoring and Evaluation")
        # Insert main data visualization
        if data:
            data = pd.read_csv(data)
            data.drop(columns=["#"],inplace=True)
            df_long,df_wide = transform_data(data)
            with cola2:
                choosen_area, choosen_school, choosen_batch, choosen_set = enable_filter(df_long,df_wide)
            df_wide,df_long = return_filtered_data(df_wide,df_long,choosen_area, choosen_school, choosen_batch, choosen_set)
            main_data_viz(df_long,df_wide)

    topic = create_histogtam(df_long,choosen_school, choosen_batch, choosen_set)
    create_scatterplot(df_long,choosen_school, choosen_batch, choosen_set,topic)

    if choosen_area!="All Areas":
        bktk_df_area_specif = df_wide[df_wide["Area"]==choosen_area]
        bktk_df_long_area_specif = df_long[df_long["Area"]==choosen_area]
        site_dependent_dash(choosen_area,bktk_df_area_specif,bktk_df_long_area_specif)

main_page()
