  # -------------------------------------------------
    # Normal records
    id_list_normal = test_df[ID_COL].values.tolist()
    del test_df[ID_COL]
    test_x = test_df.values
    scores_1 = model.score_samples(test_x)

    # -------------------------------------------------
    # Positive anomalies

    anomalies_src_path = './../generated_data_v1/generated_anomalies/{}'.format(DIR)
    test_df_p = pd.read_csv(os.path.join(anomalies_src_path, 'pos_anomalies.csv' ), index_col=None)
    id_list_p = test_df_p[ID_COL].values.tolist()
    del test_df_p[ID_COL]
    test_xp = test_df_p.values
    scores_2 =  model.score_samples(test_xp)

    # -------------------------------------------------
    # Negative anomalies
    
    test_df_n = pd.read_csv(os.path.join(anomalies_src_path, 'neg_anomalies.csv' ), index_col=None)
    id_list_n = test_df_n[ID_COL].values.tolist()
    del test_df_n[ID_COL]
    test_xn = test_df_n.values
    scores_3 = model.score_samples(test_xn)

    try:
        box = plt.boxplot([scores_1,scores_2,scores_3], notch=True, patch_artist=True)
        colors = ['cyan', 'pink', 'lightgreen']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        plt.show()
    except:
        pass

    label_list_normal = [0 for _ in range(len(scores_1))]
    label_list_p = [1 for _ in range(len(scores_2))]
    label_list_n = [-1 for _ in range(len(scores_3))]
    scores = scores_1 + scores_2 + scores_3
    id_list = id_list_normal + id_list_p + id_list_n
    labels = label_list_normal + label_list_p + label_list_n
    data = {'label': labels, 'score': scores , 'PanjivaRecordID': id_list}
    df = pd.DataFrame(data)