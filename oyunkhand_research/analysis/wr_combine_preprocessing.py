# 1A. Файлуудыг унших, бүтэц шалгах, WR тоглогчдыг сонгох
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

SEED = 2027
rng = np.random.default_rng(SEED)
# Colab-д файлууд /content дотор байна. Өөр газар бол NFL_DATA_DIR-ийг тохируулна.
DATA_DIR = Path(os.environ.get("NFL_DATA_DIR", "/content"))
OUTPUT_DIR = DATA_DIR / "wr_preprocessing_outputs"
REQUIRED = {
    "players": ("players.csv", ["nfl_id", "display_name", "draft_year", "nfl_position"]),
    "combine_results": ("combine_results.csv", ["nfl_id", "draft_year", "forty"]),
    "combine_tracking": ("combine_tracking.csv", [
        "draft_year", "event_id", "nfl_id", "entity_type", "time", "drill_type",
        "drill_name", "attempt", "x", "y", "s", "a", "dis", "dir"
    ]),
}
missing_files = [filename for filename, _ in REQUIRED.values()
                 if not (DATA_DIR / filename).is_file()]
if missing_files:
    raise FileNotFoundError(f"{DATA_DIR} дотор файл дутуу: {missing_files}")

frames = {}
for name, (filename, required_columns) in REQUIRED.items():
    df = pd.read_csv(DATA_DIR / filename,
                     dtype={"event_id": "string"} if name == "combine_tracking" else None)
    absent = set(required_columns) - set(df.columns)
    if absent:
        raise ValueError(f"{filename}: багана дутуу: {sorted(absent)}")
    frames[name] = df
players = frames["players"]
combine_results = frames["combine_results"]
combine_tracking = frames["combine_tracking"]

for name, df in [("players", players), ("combine_results", combine_results)]:
    if df[["nfl_id", "draft_year"]].isna().any().any():
        raise ValueError(f"{name}: тоглогч/жилийн түлхүүрт дутуу утга байна")
    if df["nfl_id"].duplicated().any():
        raise ValueError(f"{name}: nfl_id давхардсан; холболтыг эхлээд засна уу")

players["nfl_position"] = players["nfl_position"].astype("string").str.strip().str.upper()
wr_players = players.loc[players["nfl_position"].eq("WR")].copy()
if wr_players.empty:
    raise ValueError("WR тоглогч олдсонгүй. Байрлалын кодыг шалгана уу.")
wr_ids = set(wr_players["nfl_id"])
wr_combine_results = combine_results.loc[combine_results["nfl_id"].isin(wr_ids)].copy()
wr_tracking = combine_tracking.loc[
    combine_tracking["entity_type"].astype("string").str.upper().eq("PLAYER")
    & combine_tracking["nfl_id"].isin(wr_ids)
].copy()
wr_tracking = wr_tracking.merge(
    wr_players[["nfl_id", "display_name", "draft_year"]].rename(
        columns={"draft_year": "player_draft_year"}),
    on="nfl_id", how="left", validate="many_to_one"
)
if not wr_tracking["draft_year"].eq(wr_tracking["player_draft_year"]).all():
    raise ValueError("Combine-ийн жил тоглогчийн draft_year-тай зөрж байна")

file_summary = pd.DataFrame([
    {"table": name, "rows": len(df), "columns": df.shape[1],
     "memory_MB": round(df.memory_usage(deep=True).sum() / 1024**2, 2)}
    for name, df in frames.items()
])
print("Уншсан файлууд:")
display(file_summary)
print(f"WR: {len(wr_players)} тоглогч | PLAYER tracking: {len(wr_tracking):,} мөр")
print("Байрлалын тархалт:")
display(players["nfl_position"].value_counts().rename_axis("position").to_frame("players"))
print("WR сорилуудын хамрах хүрээ:")
display(wr_tracking.groupby("drill_type").agg(
    players=("nfl_id", "nunique"), events=("event_id", "nunique"),
    frames=("nfl_id", "size")
).sort_values("players", ascending=False))
print("WR tracking-ийн дутуу утга (%):")
display(wr_tracking[REQUIRED["combine_tracking"][1]].isna().mean().mul(100)
        .round(2).to_frame("missing_pct"))
print("Тайлбар: frame-ийн тоо нь бие даасан тоглогчийн түүврийн хэмжээ биш.")


# 1B. Дараалсан координатаас эргэлтийн өнцөг гаргах
# Өгөгдсөн s = хурд. step_length_yards = frame хоорондын шилжилт; эдгээр өөр хэмжигдэхүүн.
EXPECTED_DT_S = 0.10
DT_TOLERANCE_S = 0.015
MIN_SPEED_YD_S = 1.0  # Эхний QC босго; дараа нь 0.5/1.0/1.5 дээр мэдрэмжийг шалгана.
MAX_SPEED_YD_S = 15.0  # Зөвхөн илт зөрүүтэй координатыг QC-д тэмдэглэх өргөн босго.
SEQ = ["nfl_id", "draft_year", "event_id", "drill_type", "drill_name", "attempt"]
KEY = SEQ + ["timestamp"]

wr_motion = wr_tracking.copy()
wr_motion["timestamp"] = pd.to_datetime(wr_motion["time"], errors="coerce", utc=True)
for column in ["x", "y", "s", "a", "dis", "dir"]:
    wr_motion[column] = pd.to_numeric(wr_motion[column], errors="coerce")
if wr_motion[SEQ + ["timestamp"]].isna().any().any():
    raise ValueError("Дарааллын түлхүүр эсвэл timestamp дутуу/буруу; эх өгөгдлийг шалгана уу")

# Ижил бичлэгийг л арилгана; ижил цагт зөрүүтэй байрлалтай бол чимээгүй сонгохгүй.
raw_rows = len(wr_motion)
wr_motion = wr_motion.drop_duplicates().copy()
if wr_motion.duplicated(KEY, keep=False).any():
    raise ValueError("Ижил тоглогч/сорил/оролдлого/цаг дээр зөрүүтэй бичлэг байна")
wr_motion = wr_motion.sort_values(KEY).reset_index(drop=True)
grouped = wr_motion.groupby(SEQ, sort=False, observed=True)
prev_x, prev_y = grouped["x"].shift(1), grouped["y"].shift(1)
next_x, next_y = grouped["x"].shift(-1), grouped["y"].shift(-1)
prev_time = grouped["timestamp"].shift(1)
next_time = grouped["timestamp"].shift(-1)
wr_motion["dt_prev_s"] = (wr_motion["timestamp"] - prev_time).dt.total_seconds()
wr_motion["dt_next_s"] = (next_time - wr_motion["timestamp"]).dt.total_seconds()

# v_in = p(t)-p(t-1), v_out = p(t+1)-p(t).
in_x, in_y = wr_motion["x"] - prev_x, wr_motion["y"] - prev_y
out_x, out_y = next_x - wr_motion["x"], next_y - wr_motion["y"]
wr_motion["step_length_yards"] = np.hypot(out_x, out_y)
wr_motion["speed_in_yd_s"] = np.hypot(in_x, in_y) / wr_motion["dt_prev_s"]
wr_motion["speed_out_yd_s"] = wr_motion["step_length_yards"] / wr_motion["dt_next_s"]
# atan2(cross, dot) нь -pi..pi завсарт тойргийн заагийг зөв тооцно.
cross = in_x * out_y - in_y * out_x
dot = in_x * out_x + in_y * out_y
wr_motion["turn_angle_raw_rad"] = np.arctan2(cross, dot)
wr_motion["qc_contiguous"] = (
    wr_motion["dt_prev_s"].sub(EXPECTED_DT_S).abs().le(DT_TOLERANCE_S)
    & wr_motion["dt_next_s"].sub(EXPECTED_DT_S).abs().le(DT_TOLERANCE_S)
)
wr_motion["qc_speed"] = (
    wr_motion["speed_in_yd_s"].between(MIN_SPEED_YD_S, MAX_SPEED_YD_S)
    & wr_motion["speed_out_yd_s"].between(MIN_SPEED_YD_S, MAX_SPEED_YD_S)
)
wr_motion["valid_turn"] = wr_motion["qc_contiguous"] & wr_motion["qc_speed"]
wr_motion["turn_angle_rad"] = wr_motion["turn_angle_raw_rad"].where(wr_motion["valid_turn"])
wr_motion["turn_angle_deg"] = np.degrees(wr_motion["turn_angle_rad"])
# Эдгээр нь сонгосон шалгуураар хүчинтэй өнцгүүд; бүх мөр wr_motion-д хэвээр.
wr_turns_all_drills = wr_motion.loc[wr_motion["valid_turn"]].copy()
# Эхний судалгааны түүвэр: WR байрлалын сорил. 40-yard/shuttle/three-cone-ийг тусад нь хадгална.
wr_skill_turns = wr_turns_all_drills.loc[
    wr_turns_all_drills["drill_type"].eq("SKILL_DRILLS_WR")
].copy()
if wr_skill_turns.empty:
    raise ValueError("QC-ийн дараа WR skill drill-ийн хүчинтэй өнцөг үлдсэнгүй")

qc_summary = pd.Series({
    "raw_WR_PLAYER_rows": raw_rows,
    "exact_duplicate_rows_removed": raw_rows - len(wr_motion),
    "rows_after_deduplication": len(wr_motion),
    "contiguous_three_point_rows": int(wr_motion["qc_contiguous"].sum()),
    "valid_turn_rows_all_drills": len(wr_turns_all_drills),
    "valid_turn_rows_WR_skill_drills": len(wr_skill_turns),
    "WR_players_with_valid_skill_turns": wr_skill_turns["nfl_id"].nunique(),
})
display(qc_summary.to_frame("count"))
print("QC-ийн дараах WR skill сорилууд:")
skill_coverage = wr_skill_turns.groupby("drill_name").agg(
    players=("nfl_id", "nunique"), events=("event_id", "nunique"),
    valid_turns=("turn_angle_rad", "size")
).sort_values("players", ascending=False)
display(skill_coverage)
print("Босго нь судалгааны эцсийн сонголт биш. Удаан хөдөлгөөний шуугиан ба тасалдлыг тэмдэглэв.")
print("Том өнцөг / их хэлбэлзэл = сайн гүйцэтгэл гэж одоогоор дүгнэхгүй.")


# 1C. Харааны шалгалт, дараагийн шатанд хэрэглэх хүснэгтүүдийг хадгалах
# Хамгийн олон тоглогчтой нэг ижил нэртэй сорилыг л эхний зурагт харуулна.
example_drill = skill_coverage.index[0]
example_turns = wr_skill_turns.loc[wr_skill_turns["drill_name"].eq(example_drill)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(example_turns["turn_angle_deg"], bins=60, color="#2864a0")
axes[0].set(xlabel="Turn angle (degrees)", ylabel="Frames", title=example_drill)
plot_sample = example_turns.sample(min(5000, len(example_turns)), random_state=SEED)
axes[1].scatter(plot_sample["speed_in_yd_s"], plot_sample["turn_angle_deg"],
                s=7, alpha=0.2, color="#2864a0")
axes[1].set(xlabel="Incoming speed (yards/second)", ylabel="Turn angle (degrees)",
            title="QC only: speed and turning")
plt.tight_layout()
plt.show()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
wr_players.to_csv(OUTPUT_DIR / "wr_players.csv", index=False)
wr_combine_results.to_csv(OUTPUT_DIR / "wr_combine_results.csv", index=False)
wr_motion.to_csv(OUTPUT_DIR / "wr_motion_qc.csv", index=False)
wr_skill_turns.to_csv(OUTPUT_DIR / "wr_skill_turns.csv", index=False)
skill_coverage.to_csv(OUTPUT_DIR / "wr_skill_coverage.csv")
qc_summary.to_csv(OUTPUT_DIR / "qc_summary.csv", header=["count"])
fig.savefig(OUTPUT_DIR / "wr_turn_qc.png", dpi=160, bbox_inches="tight")
print(f"Хадгалсан: {OUTPUT_DIR}")
print("Дараагийн шат: сорилын хэсгийг ялгах; хурд, сорилын хэлбэрийг тооцсон circular mixed-effects загвар.")
print("Анхаарах: Colab runtime устахад /content-ийн файлууд устаж болно. Notebook-ийн код Drive-д хадгалагдана.")
