
def prepare_fungus_data(df_ori):
    df = df_ori[df_ori["类别"] == "fungus"]

    df = df.drop_duplicates(subset=["序列", "作用位点"])
    df = df[df["作用位点"].str.contains("Lipid Bilayer", na=True)]
    df = df.reset_index(drop=True)
    return df