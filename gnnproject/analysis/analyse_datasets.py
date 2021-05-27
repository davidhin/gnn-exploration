# %%
import json
from collections import Counter
from glob import glob
from pathlib import Path

import gnnproject as gp
import pandas as pd

# %% ----------------------------------------------------------------- #
#                              ReVeal                                  #
# -------------------------------------------------------------------- #
reveal = glob(str(gp.external_dir() / "reveal_chrome_debian" / "functions" / "*"))
reveal_vul = [Path(i).stem.split("_")[1] for i in reveal]
Counter(reveal_vul)


# %% ----------------------------------------------------------------- #
#                              DEVIGN                                  #
# -------------------------------------------------------------------- #

with open(gp.external_dir() / "function.json") as f:
    j = json.load(f)

df = pd.DataFrame(j).sort_values("target")
sample = df[df.target == 1].sample(20, random_state=0)
sample.func = sample.func.apply(lambda x: x.replace("\n", "")[:150])
sample.to_csv(gp.external_dir() / "devign_ffmpeg_qemu" / "class_mappings.csv", index=0)

df.groupby("target").count()
df[df.commit_id == "ad2d30f79d3b0812f02c741be2189796b788d6d7"]

# %% ----------------------------------------------------------------- #
#                              BIG-VUL                                  #
# -------------------------------------------------------------------- #
msr2020 = pd.read_csv(
    "/media/david/DAVIDH/ReVealData/MSR2020Dataset/MSR_data_cleaned.csv"
)
msr2020[(msr2020.vul == 0) & (msr2020.lang == "C")]

# %% ----------------------------------------------------------------- #
#                              DRAPER                                  #
# -------------------------------------------------------------------- #

drap_val = pd.read_parquet(gp.external_dir() / "draper/draper_validate.parquet")
drap_test = pd.read_parquet(gp.external_dir() / "draper/draper_test.parquet")
drap_train = pd.read_parquet(gp.external_dir() / "draper/draper_train.parquet")


drap_val["vul"] = drap_val.apply(lambda x: len([i for i in x if i == True]) > 0, axis=1)
drap_test["vul"] = drap_test.apply(
    lambda x: len([i for i in x if i == True]) > 0, axis=1
)
drap_train["vul"] = drap_train.apply(
    lambda x: len([i for i in x if i == True]) > 0, axis=1
)

pd.concat([drap_val, drap_test, drap_train]).groupby("vul").count()

# %% ----------------------------------------------------------------- #
#                                 D2A                                  #
# -------------------------------------------------------------------- #
d2a = pd.read_csv("/media/david/DAVIDH/ReVealData/D2A/d2a/splits.csv")
d2a["vul"] = d2a.id.apply(lambda x: x.split("_")[-1])
d2a.groupby("vul").count()
