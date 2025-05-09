import streamlit as st
import pandas as pd
from scipy.stats import wilcoxon
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
        page_title="BKTK Angat Buhay",
        
    )
st.title("ANGAT BUHAY")
st.write("### Bayan Ko Titser Ko (BKTK)")

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

    # st.plotly_chart(fig)


    st.write(bktk_agg_trans)

    sortHow = st.selectbox(
        "How do you want to sort your data?",
        ["None","Ascending","Descending"],
        index=None,
        placeholder="Select topic ...",
        )    
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
    if sortHow == "Ascending":
            sort_by = True
    else:
            sort_by = False
    meanDf["Topics"] = meanDf["Topics"].str.replace(r"\s?\(?(Pretest|Posttest)\)?", "", regex=True)

    if sortHow == "None":
            usedData_mean_pre = meanDf[meanDf["Test Type"]=="Pretest"]
            usedData_mean_post = meanDf[meanDf["Test Type"]=="Posttest"]
    else:
            usedData_mean_pre = meanDf[meanDf["Test Type"]=="Pretest"].sort_values(by="Mean Values",ascending=sort_by)
            usedData_mean_post = meanDf[meanDf["Test Type"]=="Posttest"].sort_values(by="Mean Values",ascending=sort_by)
    fig_mean_pre = px.bar(usedData_mean_pre,
                      x="Topics",
                      y="Mean Values",
                      color="Test Type",
                      facet_col="Test Type",
                      title="Mean Values by Topic (Pretest)",
                      color_discrete_map={"Pretest": "lightblue"},
                      text="Mean Values"
                     )
    fig_mean_pre.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_mean_pre.update_yaxes(range=[0, 105])

    fig_mean_post = px.bar(usedData_mean_post,
                      x="Topics",
                      y="Mean Values",
                      color="Test Type",
                      facet_col="Test Type",
                      title="Mean Values by Topic (Posttest)",
                      color_discrete_map={"Posttest": "pink"},
                      text="Mean Values"
                     )
    fig_mean_post.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_mean_post.update_yaxes(range=[0, 105])
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_mean_pre)

    with col2:
        st.plotly_chart(fig_mean_post)

    meanPivot = meanDf.pivot(index="Topics", columns="Test Type", values="Mean Values").reset_index()
    meanPivot["Difference"] = meanPivot["Posttest"] - meanPivot["Pretest"]

    if sortHow == "None":
         diffUsed = meanPivot
    else:
         diffUsed = meanPivot.sort_values(by="Difference",ascending=sort_by)

    fig_mean_dif = px.bar(diffUsed,
                      x="Topics",
                      y="Difference",
                      title="Mean Difference Values by Topic",
                      text="Difference",
                     )
    fig_mean_dif.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_mean_dif.update_yaxes(range=[0, 105])
    st.plotly_chart(fig_mean_dif)


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
    
    if sortHow == "None":
            usedData_cv_pre = cvDf[cvDf["Test Type"]=="Pretest"]
            usedData_cv_post = cvDf[cvDf["Test Type"]=="Posttest"]
    else:
            usedData_cv_pre = cvDf[cvDf["Test Type"]=="Pretest"].sort_values(by="Coefficent of Variation Values",ascending=sort_by)
            usedData_cv_post = cvDf[cvDf["Test Type"]=="Posttest"].sort_values(by="Coefficent of Variation Values",ascending=sort_by)
    fig_cv_pre = px.bar(usedData_cv_pre,
                      x="Topics",
                      y="Coefficent of Variation Values",
                      color="Test Type",
                      facet_col="Test Type",
                      title="Coefficent of Variation Values by Topic (Pretest)",
                      color_discrete_map={"Pretest": "lightblue"},
                      text="Coefficent of Variation Values"
                     )
    fig_cv_pre.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_cv_pre.update_yaxes(range=[0,2.5])
    fig_cv_post = px.bar(usedData_cv_post,
                      x="Topics",
                      y="Coefficent of Variation Values",
                      color="Test Type",
                      facet_col="Test Type",
                      title="Coefficent of Variation Values by Topic (Posttest)",
                      text="Coefficent of Variation Values",
                      color_discrete_map={"Posttest": "pink"}
                     )
    fig_cv_post.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_cv_post.update_yaxes(range=[0,2.5])
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_cv_pre)

    with col2:
        st.plotly_chart(fig_cv_post)


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

    # stat, p_value = wilcoxon(preTestScore, postTestScore)

    # stat, p = wilcoxon(preTestScore, postTestScore)

    #  n = len(preTestScore)
    #  mean_W = n * (n + 1) / 4
    #  std_W = (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
    #  z = (stat - mean_W) / std_W
     
    #  effect_size_r = abs(z) / np.sqrt(n)

    fig_hist_gen = px.histogram(
        filtered_bktk_df,
        x="Score",
        nbins=bin_num,
        title=f"{topic} Score Distribution (Pretest vs Posttest)",
        facet_col="Test Type",
        color="Test Type",
        color_discrete_map={"Pretest": "lightblue", "Posttest": "pink"}
    )

    st.write(f"Wilxon Signed Rank Test P-Value: ",p_value)
    # st.write(f"Effect Size: ",round(effect_size_r,5))

    st.plotly_chart(fig_hist_gen)

    st.title("Z-Scores")

    long_bktk_df_zscore = long_bktk_df.copy()
    for typeIter in list(long_bktk_df_zscore["Test Type"].unique()):
          for topicIter in list(long_bktk_df_zscore["Topics"].unique()):
                subset = long_bktk_df_zscore[
            (long_bktk_df_zscore["Topics"] == topicIter) & 
            (long_bktk_df_zscore["Test Type"] == typeIter)]
                subset_mean = subset["Score"].mean()
                subset_std = subset["Score"].std(ddof=0)

                long_bktk_df_zscore.loc[(long_bktk_df_zscore["Topics"] == topicIter) & (long_bktk_df_zscore["Test Type"] == typeIter), "Score"] = (subset["Score"] - subset_mean) / subset_std
    topicForZScore = st.selectbox(
        "What topic do you want to explore the Z-score of?",
        list(long_bktk_df["Topics"].unique()),
        index=None,
        placeholder="Select topic ...",
        )
    st.write("You have chosen topic: ",topicForZScore)

    fig_scat = px.scatter(
        long_bktk_df_zscore[long_bktk_df_zscore["Topics"]==topicForZScore],
        x="Name of Child",
        y="Score",
        title=f"{topicForZScore} Z-Score Distribution (Pretest vs Posttest)",
        color="Test Type",
        color_discrete_map={"Pretest": "lightblue", "Posttest": "pink"}
    )
    fig_scat.update_layout(
        height=600
    )
    st.plotly_chart(fig_scat)



data = st.file_uploader(label="Upload your .csv file")
if data:
    df = pd.read_csv(data)
    deployDash(df)
