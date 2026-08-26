import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import datetime

warnings.filterwarnings("ignore")

st.set_page_config(page_title="电力交易用电量预测平台", page_icon="⚡", layout="wide")

# ==============================================================================
# 【全局配置数据】
# ==============================================================================
USER_EXPANSION = {
    1:  {"BD新增": 0,    "BD总用电量": 29739968, "BD均值": 3141, "10KV新增": 0,  "10KV总用电量": 23854477, "10KV新增用电量": 0},
    2:  {"BD新增": 100,  "BD总用电量": 22440043, "BD均值": 2345, "10KV新增": 1,  "10KV总用电量": 12377987, "10KV新增用电量": 196488},
    3:  {"BD新增": 41,   "BD总用电量": 23196875, "BD均值": 2414, "10KV新增": 1,  "10KV总用电量": 23724756, "10KV新增用电量": 87492},
    4:  {"BD新增": 1651, "BD总用电量": 25836521, "BD均值": 2294, "10KV新增": 0,  "10KV总用电量": 24106814, "10KV新增用电量": 0},
    5:  {"BD新增": 844,  "BD总用电量": 32096484, "BD均值": 2651, "10KV新增": 2,  "10KV总用电量": 23384094, "10KV新增用电量": 115913},
    6:  {"BD新增": 1131, "BD总用电量": 35953341, "BD均值": 2716,  "10KV新增": 10, "10KV总用电量": 23438111, "10KV新增用电量": 4534000},
    7:  {"BD新增": 1363, "BD总用电量": 59389939, "BD均值": 2594, "10KV新增": 12, "10KV总用电量": 26434314, "10KV新增用电量": 7190000},
    8:  {"BD新增": 1464, "BD总用电量": None,      "BD均值": None,  "10KV新增": 11, "10KV总用电量": None,      "10KV新增用电量": 2090000},
    9:  None, 10: None, 11: None, 12: None,
}

MONTHLY_ELECTRICITY_CURVE = {
    1: 0.07953251, 2: 0.07301489, 3: 0.07049988, 4: 0.06691741,
    5: 0.07716002, 6: 0.09088047, 7: 0.12436856, 8: 0.12395636,
    9: 0.07984493, 10: 0.07016185, 11: 0.06719386, 12: 0.07646927
}

JULY_SHAPE_WEEKDAY = [0.03193, 0.03702, 0.03435, 0.03152, 0.02880, 0.02763, 0.02821, 0.03000, 0.03825, 0.04638, 0.04910, 0.04958, 0.04853, 0.04923, 0.04927, 0.04956, 0.05090, 0.05269, 0.05402, 0.05242, 0.04835, 0.04167, 0.03557, 0.03503]
JULY_SHAPE_WEEKEND = [0.03223, 0.03759, 0.03467, 0.03131, 0.02922, 0.02781, 0.02804, 0.02926, 0.03706, 0.04520, 0.04853, 0.04933, 0.04907, 0.04965, 0.04969, 0.05005, 0.05134, 0.05320, 0.05416, 0.05242, 0.04855, 0.04167, 0.03540, 0.03456]

TEMP_CITIES = ["达州_体感温度(℃)", "泸州_体感温度(℃)", "南充_体感温度(℃)", "绵阳_体感温度(℃)", "眉山_体感温度(℃)", "内江_体感温度(℃)", "宜宾_体感温度(℃)", "广安_体感温度(℃)"]

CHINESE_HOLIDAYS_2026 = {"2026-01-01", "2026-01-02", "2026-01-03", "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20", "2026-02-21", "2026-02-22", "2026-04-04", "2026-04-05", "2026-04-06", "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05", "2026-06-19", "2026-06-20", "2026-06-21", "2026-09-25", "2026-09-26", "2026-09-27", "2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07"}

TEMP_ELASTICITY_CAP = 0.04
TEMP_EXCESS_CAP = 1.5
TEMP_CORRECTION_FACTOR = 0.4
COOL_DOWN_THRESHOLD = 5.0
COOL_DOWN_DAY_FACTOR = 0.01
COOL_DOWN_EVENING_FACTOR = 0.015
COOL_DOWN_EARLY_FACTOR = 0.005

# ==============================================================================
# 【特征工程函数】
# ==============================================================================
def build_advanced_features(df_raw, last_actual_user_date, last_actual_system_date):
    df = df_raw.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Hour"] = df["Hour"].astype(int)
    df["Month"] = df["Date"].dt.month
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = df["DayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)
    df["DayOfMonth"] = df["Date"].dt.day

    df["Is_Chinese_Holiday"] = (df["Date"].dt.strftime("%Y-%m-%d").isin(CHINESE_HOLIDAYS_2026).astype(int))
    df["Day_Type"] = 0
    df.loc[df["IsWeekend"] == 1, "Day_Type"] = 1
    df.loc[df["Is_Chinese_Holiday"] == 1, "Day_Type"] = 2

    df["Days_After_Holiday"] = 0
    df["Days_Before_Holiday"] = 0
    holiday_dt_set = set(pd.to_datetime(list(CHINESE_HOLIDAYS_2026)))
    for n in range(1, 6):
        mask = df["Date"].apply(lambda x: (x - pd.Timedelta(days=n)) in holiday_dt_set)
        df.loc[mask & (df["Days_After_Holiday"] == 0), "Days_After_Holiday"] = n
    for n in range(1, 4):
        mask = df["Date"].apply(lambda x: (x + pd.Timedelta(days=n)) in holiday_dt_set)
        df.loc[mask & (df["Days_Before_Holiday"] == 0), "Days_Before_Holiday"] = n
    df["Is_Post_Holiday"] = (df["Days_After_Holiday"].isin([1, 2])).astype(int)
    df["Is_Pre_Holiday"] = (df["Days_Before_Holiday"] > 0).astype(int)

    df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24.0)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24.0)
    df["Is_Month_Start"] = (df["DayOfMonth"] <= 7).astype(int)
    df["Is_Month_End"] = (df["DayOfMonth"] >= 25).astype(int)
    df["Is_Quarter_End"] = (df["Month"].isin([3, 6, 9, 12]) & (df["DayOfMonth"] >= 25)).astype(int)
    df["Is_Daytime_Peak_Hours"] = ((df["Hour"] >= 9) & (df["Hour"] <= 19)).astype(int)

    # K_load校准
    sys_cutoff = pd.to_datetime(last_actual_system_date)
    mon_actual = sys_cutoff - pd.Timedelta(days=sys_cutoff.weekday())
    sun_actual = mon_actual + pd.Timedelta(days=6)
    calib_mask = (df["Date"] >= mon_actual) & (df["Date"] <= sys_cutoff)
    act_sum = df.loc[calib_mask, "实际负荷"].sum()
    fore_sum = df.loc[calib_mask, "预计负荷"].sum()
    K_load = (act_sum / fore_sum) if (fore_sum > 0 and pd.notna(act_sum)) else 1.0
    mask_gap = (df["Date"] > sys_cutoff) & (df["Date"] <= sun_actual)
    df["预计负荷_校准"] = np.where(mask_gap, df["预计负荷"] * K_load, df["预计负荷"])

    mask_sys_act = df["Date"] <= sys_cutoff
    df["Effective_Load"] = np.where(mask_sys_act, df["实际负荷"], df["预计负荷_校准"])
    df["Effective_Hydro"] = np.where(mask_sys_act, df["水电总出力"], df["预计水电"])
    df["Effective_Renewable"] = np.where(mask_sys_act, df["新能源总出力"], df["预计新能源"])

    df["竞价空间"] = df["Effective_Load"] - df["Effective_Hydro"] - df["Effective_Renewable"]
    df["非市场化机组总出力"] = pd.to_numeric(df["非市场化机组总出力"], errors="coerce")
    df["真实竞价空间"] = df["竞价空间"] - df["非市场化机组总出力"]

    # 气象
    valid_temps = [c for c in TEMP_CITIES if c in df.columns]
    if valid_temps:
        for c in valid_temps:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["全省最高体感"] = df[valid_temps].max(axis=1)
        daily_max = df.groupby("Date")[valid_temps].transform("max").max(axis=1)
        daily_min = df.groupby("Date")[valid_temps].transform("min").min(axis=1)
        df["气温日较差"] = daily_max - daily_min
        df["CoolingDegree"] = df["全省最高体感"].apply(lambda x: max(x - 26.0, 0.0))

        daily_hot = df.groupby("Date")["全省最高体感"].max() >= 35.0
        hot_counts, count = [], 0
        for is_h in daily_hot:
            count = count + 1 if is_h else 0
            hot_counts.append(count)
        df["Consecutive_Hot_Days"] = df["Date"].map(dict(zip(daily_hot.index, hot_counts)))
        df["Consecutive_Hot_Days_Sq"] = df["Consecutive_Hot_Days"] ** 2

        daily_max_series = df.groupby("Date")["全省最高体感"].max()
        df["Temp_Mean_3D"] = df["Date"].map(daily_max_series.rolling(3, min_periods=1).mean())
        df["Temp_Mean_5D"] = df["Date"].map(daily_max_series.rolling(5, min_periods=1).mean())
        df["Temp_Mean_7D"] = df["Date"].map(daily_max_series.rolling(7, min_periods=1).mean())
        df["Temp_Mean_10D"] = df["Date"].map(daily_max_series.rolling(10, min_periods=1).mean())
        df["Weekend_Heat_Surge"] = df["IsWeekend"] * df["Temp_Mean_5D"]

        df["Is_Extreme_Heat"] = (df["CoolingDegree"] > 7).astype(int)
        df["Extreme_Heat_Hour"] = df["Is_Extreme_Heat"] * df["Hour_Sin"]
        df["Extreme_Heat_Hour_Cos"] = df["Is_Extreme_Heat"] * df["Hour_Cos"]
        df["Heat_Accumulation_3D"] = df["Consecutive_Hot_Days"].clip(upper=5) * df["Temp_Mean_3D"]
        df["Temp_Prev3h_Mean"] = df.groupby("Date")["CoolingDegree"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        df["Temp_Decay_From_Peak"] = df["Temp_Prev3h_Mean"] - df["CoolingDegree"]
        df["Cooling_Bin"] = np.digitize(df["CoolingDegree"], [0, 3, 6, 10])
        df["Cooling_Hour_Peak"] = df["CoolingDegree"] * df["Is_Daytime_Peak_Hours"]
        df["CoolingDegree_Sq"] = df["CoolingDegree"] ** 2
        df["CoolingDegree_x3"] = df["CoolingDegree"] ** 3
        df["Temp_vs_3D_Ratio"] = df["CoolingDegree"] / (df["Temp_Mean_3D"] + 1e-5)
        df["Temp_vs_5D_Ratio"] = df["CoolingDegree"] / (df["Temp_Mean_5D"] + 1e-5)
        df["Cooling_Load_Interaction"] = df["CoolingDegree"] * df["Effective_Load"] / 10000
        df["Cooling_Drop_24h"] = df.groupby("Hour")["CoolingDegree"].diff() * -1
    else:
        for c in ["全省最高体感", "气温日较差", "CoolingDegree", "Consecutive_Hot_Days", "Temp_Mean_3D", "Temp_Mean_5D", "Temp_Mean_7D", "Temp_Mean_10D", "CoolingDegree_Sq", "CoolingDegree_x3", "Temp_vs_3D_Ratio", "Temp_vs_5D_Ratio", "Cooling_Load_Interaction", "Cooling_Drop_24h"]:
            df[c] = 0.0

    # 月度用户特征
    df["BD_New_Count"] = df["Month"].map(lambda m: USER_EXPANSION.get(m, {}).get("BD新增", 0) if USER_EXPANSION.get(m) else 0)
    df["BD_Total_Electricity"] = df["Month"].map(lambda m: USER_EXPANSION.get(m, {}).get("BD总用电量", 0) if USER_EXPANSION.get(m) else 0)
    df["BD_Avg_Electricity"] = df["Month"].map(lambda m: USER_EXPANSION.get(m, {}).get("BD均值", 0) if USER_EXPANSION.get(m) else 0)
    df["V10KV_New_Count"] = df["Month"].map(lambda m: USER_EXPANSION.get(m, {}).get("10KV新增", 0) if USER_EXPANSION.get(m) else 0)
    df["V10KV_New_Electricity"] = df["Month"].map(lambda m: USER_EXPANSION.get(m, {}).get("10KV新增用电量", 0) if USER_EXPANSION.get(m) else 0)

    df["Hour_CoolingDegree"] = df["Hour"] * df["CoolingDegree"]
    df["Hour_Weekend"] = df["Hour"] * 2 + df["IsWeekend"]
    df["Temp_Diff_24h"] = df["全省最高体感"] - df["全省最高体感"].shift(24)
    df["Temp_Diff_168h"] = df["全省最高体感"] - df["全省最高体感"].shift(168)

    df["N_10kV_new"] = pd.to_numeric(df["N_10kV_new"], errors="coerce").fillna(0)
    df["10KV_history_amount"] = pd.to_numeric(df["10KV_history_amount"], errors="coerce").fillna(0)
    df["Industrial_Growth_Index"] = df["N_10kV_new"] * df["10KV_history_amount"]

    df["负荷变幅_24h"] = df["Effective_Load"] - df["Effective_Load"].shift(24)
    df["负荷变幅_1h"] = df["Effective_Load"] - df["Effective_Load"].shift(1)

    df["Load_Lag24"] = df["Effective_Load"].shift(24)
    df["Load_Lag48"] = df["Effective_Load"].shift(48)
    df["Load_Lag72"] = df["Effective_Load"].shift(72)
    df["Load_Lag168"] = df["Effective_Load"].shift(168)
    df["Load_Ratio168"] = df["Effective_Load"] / (df["Load_Lag168"] + 1e-5)
    df["Load_Ratio24"] = df["Effective_Load"] / (df["Load_Lag24"] + 1e-5)

    # 电量滞后
    lookback_days = 3
    cutoff_date = pd.to_datetime(last_actual_user_date)
    start_date = cutoff_date - pd.Timedelta(days=lookback_days - 1)
    lookback_mask = ((df["Date"] >= start_date) & (df["Date"] <= cutoff_date) & df["实际电量"].notna() & df["实际负荷"].notna() & (df["实际负荷"] > 0))
    default_ratio = (df.loc[lookback_mask, "实际电量"].sum() / df.loc[lookback_mask, "实际负荷"].sum()) if lookback_mask.sum() > 0 else 0.05
    hourly_ratio = {h: default_ratio for h in range(1, 25)}
    for hour in range(1, 25):
        hour_mask = lookback_mask & (df["Hour"] == hour)
        if hour_mask.sum() > 0:
            es = df.loc[hour_mask, "实际电量"].sum()
            ls = df.loc[hour_mask, "实际负荷"].sum()
            if ls > 0:
                hourly_ratio[hour] = es / ls
    df["Hourly_Ratio"] = df["Hour"].map(hourly_ratio)
    for lag in [24, 48, 72, 168]:
        actual_lag = df["实际电量"].shift(lag)
        load_lag = df["Effective_Load"].shift(lag)
        ratio_lag = df["Hourly_Ratio"].shift(lag)
        df[f"elec_Lag{lag}"] = np.where(actual_lag.notna(), actual_lag, load_lag * ratio_lag)

    df["实际电量"] = pd.to_numeric(df["实际电量"], errors="coerce")
    return df


# ==============================================================================
# 【U_base构建】
# ==============================================================================
def build_u_base_smart(df, last_actual_date, target_start, target_end):
    cutoff = pd.to_datetime(last_actual_date)
    target_start_dt = pd.to_datetime(target_start)
    target_end_dt = pd.to_datetime(target_end)
    is_cross_month = (target_start_dt.month != target_end_dt.month)

    cutoff_month = cutoff.month
    search_start = cutoff.replace(day=1)
    hist = df[(df["Date"] >= search_start) & (df["Date"] <= cutoff) & df["实际电量"].notna()].copy()

    month_days = hist["Date"].nunique()
    wkdy_days = hist[hist["Day_Type"] == 0]["Date"].nunique()
    wknd_days = hist[hist["Day_Type"] == 1]["Date"].nunique()

    strategy = "distinguish"
    source = hist.copy()

    if not is_cross_month:
        if month_days >= 2 and (wkdy_days < 3 or wknd_days < 2):
            strategy = "unified"
        elif wkdy_days >= 3 and wknd_days >= 2:
            strategy = "distinguish"
        else:
            strategy = "fallback"
            prev_month = cutoff_month - 1 if cutoff_month > 1 else 12
            prev_year = cutoff.year if cutoff_month > 1 else cutoff.year - 1
            prev_mask = ((df["Date"].dt.year == prev_year) & (df["Date"].dt.month == prev_month) & df["实际电量"].notna())
            source = df[prev_mask].copy()
    else:
        if month_days == 0:
            strategy = "fallback"
            prev_month = cutoff_month - 1 if cutoff_month > 1 else 12
            prev_year = cutoff.year if cutoff_month > 1 else cutoff.year - 1
            prev_mask = ((df["Date"].dt.year == prev_year) & (df["Date"].dt.month == prev_month) & df["实际电量"].notna())
            source = df[prev_mask].copy()

    daily_temp = source.groupby("Date")["CoolingDegree"].mean()
    lookup = {}

    def get_temp_weighted_base(hour_data, target_temp, top_n=5):
        if len(hour_data) == 0:
            return None
        dates = hour_data["Date"].unique()
        daily_hour_mean = hour_data.groupby("Date")["实际电量"].mean()
        date_temps = {d: daily_temp.get(d, 0) for d in dates}
        temp_diff = {d: abs(t - target_temp) for d, t in date_temps.items()}
        sorted_dates = sorted(temp_diff.keys(), key=lambda d: temp_diff[d])
        top_dates = sorted_dates[:top_n]
        if len(top_dates) == 0:
            return None
        weights, vals = [], []
        for d in top_dates:
            diff = temp_diff[d]
            weight = np.exp(-diff / 2.0)
            vals.append(daily_hour_mean.get(d, 0))
            weights.append(weight)
        weights = np.array(weights)
        vals = np.array(vals)
        if weights.sum() > 0:
            return np.average(vals, weights=weights)
        return vals.mean()

    pred_dates_range = pd.date_range(target_start_dt, target_end_dt)
    pred_temps = []
    for pd_date in pred_dates_range:
        day_data = df[df["Date"] == pd_date]
        if len(day_data) > 0:
            pred_temps.append(day_data["CoolingDegree"].mean())
    avg_pred_temp = np.mean(pred_temps) if pred_temps else daily_temp.mean()

    for h in range(1, 25):
        hour_data = source[source["Hour"] == h]
        if strategy == "unified":
            val = get_temp_weighted_base(hour_data, avg_pred_temp, top_n=5)
            if val is not None:
                for dt in [0, 1, 2]:
                    lookup[(h, dt)] = val
        else:
            holo = hour_data[hour_data["Day_Type"] == 2]
            if len(holo) > 0:
                val = get_temp_weighted_base(holo, avg_pred_temp, top_n=3)
                if val is not None:
                    lookup[(h, 2)] = val
            wknd = hour_data[hour_data["Day_Type"] == 1]
            if len(wknd) > 0:
                val = get_temp_weighted_base(wknd, avg_pred_temp, top_n=5)
                if val is not None:
                    lookup[(h, 1)] = val
            wkdy = hour_data[hour_data["Day_Type"] == 0]
            if len(wkdy) > 0:
                val = get_temp_weighted_base(wkdy, avg_pred_temp, top_n=5)
                if val is not None:
                    lookup[(h, 0)] = val

    def get_u_base(row):
        h, dt = row["Hour"], row["Day_Type"]
        key = (h, dt)
        if strategy == "unified":
            return lookup.get((h, 0), 0)
        if dt == 2:
            if key in lookup:
                return lookup[key]
            base = lookup.get((h, 1), lookup.get((h, 0), 0))
            if h <= 8:
                return base * 0.82
            elif h <= 18:
                return base * 0.95
            else:
                return base * 0.90
        return lookup.get(key, 0)
    return get_u_base


# ==============================================================================
# 【扩容系数】
# ==============================================================================
def calculate_k_expansion(last_actual_user_date, target_month):
    old_m = pd.to_datetime(last_actual_user_date).month
    new_m = target_month
    old_data = USER_EXPANSION.get(old_m)
    new_data = USER_EXPANSION.get(new_m)
    if old_data is None or new_data is None:
        return 1.0
    old_total = (old_data.get("BD总用电量", 0) or 0) + (old_data.get("10KV总用电量", 0) or 0)
    old_ratio = MONTHLY_ELECTRICITY_CURVE.get(old_m, 0.09)
    new_ratio = MONTHLY_ELECTRICITY_CURVE.get(new_m, 0.09)
    month_ratio = new_ratio / old_ratio
    bd_new = (new_data.get("BD新增", 0) or 0) * (new_data.get("BD均值") or old_data.get("BD均值") or 0) * month_ratio
    v10_new = (new_data.get("10KV新增用电量", 0) or 0) / 12.0 * month_ratio
    total_new = bd_new + v10_new
    return 1.0 + (total_new / old_total) if old_total > 0 else 1.0


# ==============================================================================
# 【主预测引擎】
# ==============================================================================
def run_master_forecast(df_input, system_start, run_date, target_start, target_end):
    last_actual_user = (pd.to_datetime(run_date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    last_actual_system = (pd.to_datetime(run_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    last_actual_month = pd.to_datetime(last_actual_user).month

    df = build_advanced_features(df_input, last_actual_user, last_actual_system)
    user_cutoff = pd.to_datetime(last_actual_user)
    target_start_dt = pd.to_datetime(target_start)
    target_end_dt = pd.to_datetime(target_end)

    get_u_base_fn = build_u_base_smart(df, last_actual_user, target_start, target_end)
    df["U_base"] = df.apply(get_u_base_fn, axis=1)
    df["Target_Y"] = df["实际电量"] / (df["U_base"] + 1e-5)
    df["Lag168_Ratio"] = df["Target_Y"].shift(168)

    feature_cols = [
        "Hour", "Day_Type", "Days_After_Holiday", "Days_Before_Holiday",
        "Is_Post_Holiday", "Is_Pre_Holiday", "Hour_Sin", "Hour_Cos",
        "DayOfMonth", "Is_Month_Start", "Is_Month_End", "Is_Quarter_End",
        "Is_Daytime_Peak_Hours", "Is_Chinese_Holiday", "DayOfWeek", "IsWeekend",
        "Hour_Weekend", "Hour_CoolingDegree",
        "Effective_Load", "Effective_Hydro", "Effective_Renewable",
        "竞价空间", "真实竞价空间", "全省最高体感", "气温日较差", "CoolingDegree",
        "CoolingDegree_Sq", "CoolingDegree_x3", "Temp_vs_3D_Ratio", "Temp_vs_5D_Ratio",
        "Cooling_Load_Interaction", "Cooling_Drop_24h",
        "Cooling_Bin", "Cooling_Hour_Peak",
        "Temp_Mean_3D", "Temp_Mean_5D", "Temp_Mean_7D", "Temp_Mean_10D",
        "Weekend_Heat_Surge", "Consecutive_Hot_Days", "Consecutive_Hot_Days_Sq",
        "Is_Extreme_Heat", "Extreme_Heat_Hour", "Extreme_Heat_Hour_Cos",
        "Heat_Accumulation_3D", "Temp_Decay_From_Peak",
        "Temp_Diff_24h", "Temp_Diff_168h", "负荷变幅_24h", "负荷变幅_1h",
        "Load_Ratio168", "Load_Ratio24",
        "BD_New_Count", "BD_Total_Electricity", "BD_Avg_Electricity",
        "V10KV_New_Count", "V10KV_New_Electricity",
        "BD_history_month_weight", "N_BD_new", "Growth_BD",
        "N_10kV_new", "Growth_10kV", "Ratio_10kV", "New_User_Ratio",
        "Industrial_Growth_Index", "Lag168_Ratio",
        "Load_Lag24", "Load_Lag48", "Load_Lag72", "Load_Lag168",
        "elec_Lag24", "elec_Lag48", "elec_Lag72", "elec_Lag168"
    ] + TEMP_CITIES
    feature_cols = [c for c in feature_cols if c in df.columns]

    train_mask = (df["Date"] >= pd.to_datetime(system_start)) & (df["Date"] <= user_cutoff) & df["Target_Y"].notna()
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, "Target_Y"]

    # 🌟 添加：过滤NaN
    valid_mask = y_train.notna() & X_train.notna().all(axis=1)
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]
    
    # 同时过滤sample_weights对应的行
    df_train_filtered = df.loc[train_mask].copy()
    df_train_filtered = df_train_filtered[valid_mask]
    
    time_diff = (df.loc[train_mask, "Date"].max() - df.loc[train_mask, "Date"]).dt.days
    sample_weights = np.exp(-time_diff / 45.0)
    high_y = y_train.quantile(0.9)
    sample_weights = sample_weights * np.where(y_train > high_y, 2.0, 1.0) * np.where(df.loc[train_mask, "CoolingDegree"] > 8, 2.5, 1.0)

    model = lgb.LGBMRegressor(
        objective="regression", metric="mae", boosting_type="gbdt",
        n_estimators=500, learning_rate=0.03, num_leaves=31,
        colsample_bytree=0.6, subsample=0.8, min_child_samples=10,
        random_state=42, verbose=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    max_cooling = df.loc[train_mask, "CoolingDegree"].max()
    ht_mask = (df.loc[train_mask, "CoolingDegree"] >= 5) & (df.loc[train_mask, "CoolingDegree"] <= max_cooling)
    if ht_mask.sum() > 30:
        corr = df.loc[train_mask, "Target_Y"].corr(df.loc[train_mask, "CoolingDegree"])
        sy = df.loc[train_mask, "Target_Y"].std()
        sc = df.loc[train_mask, "CoolingDegree"].std()
        elasticity = corr * sy / sc if (sc > 0 and not np.isnan(corr) and corr > 0) else 0.02
    else:
        elasticity = 0.02
    elasticity = min(elasticity, TEMP_ELASTICITY_CAP) * TEMP_CORRECTION_FACTOR

    pred_dates = pd.date_range(target_start_dt, target_end_dt)
    for pred_date in pred_dates:
        mask_today = df["Date"] == pred_date
        df["Lag168_Ratio"] = df["Target_Y"].shift(168)
        X_today = df.loc[mask_today, feature_cols]
        if len(X_today) == 0:
            continue
        pred_today = model.predict(X_today)
        cooling = df.loc[mask_today, "CoolingDegree"].values
        for i, cd in enumerate(cooling):
            if cd > max_cooling:
                excess = min(cd - max_cooling, TEMP_EXCESS_CAP)
                if excess > 1.0:
                    pred_today[i] += (excess - 1.0) * elasticity
        df.loc[mask_today, "Pred_Ratio"] = pred_today
        df.loc[mask_today, "Working_Electricity"] = pred_today * df.loc[mask_today, "U_base"]
        df.loc[mask_today, "Target_Y"] = pred_today

    K_expansion = calculate_k_expansion(last_actual_user, target_end_dt.month)
    df["Pred_Daily_Sum"] = df.groupby("Date")["Working_Electricity"].transform("sum")

    def apply_final(row):
        m, h, dt, days_after = row["Date"].month, row["Hour"], row["Day_Type"], row["Days_After_Holiday"]
        raw = row["Working_Electricity"]
        daily_sum = row["Pred_Daily_Sum"]
        cooling_drop = row.get("Cooling_Drop_24h", 0)
        if m > last_actual_month:
            total = daily_sum * K_expansion
            shape = JULY_SHAPE_WEEKEND if dt in [1, 2] else JULY_SHAPE_WEEKDAY
            result = total * shape[h - 1]
        else:
            result = raw
        if cooling_drop > COOL_DOWN_THRESHOLD:
            drop = min(cooling_drop - COOL_DOWN_THRESHOLD, 5.0)
            if h <= 8:
                result *= (1.0 - drop * COOL_DOWN_EARLY_FACTOR)
            elif h <= 18:
                result *= (1.0 - drop * COOL_DOWN_DAY_FACTOR)
            else:
                result *= (1.0 - drop * COOL_DOWN_EVENING_FACTOR)
        if days_after == 1:
            result *= 0.95
        return result

    pred_mask = (df["Date"] >= target_start_dt) & (df["Date"] <= target_end_dt)
    df.loc[pred_mask, "预测用户用电量"] = df.loc[pred_mask].apply(apply_final, axis=1)

    df["预测误差"] = df["预测用户用电量"] - df["实际电量"]
    res = df.loc[pred_mask, ["Date", "Hour", "Day_Type", "Effective_Load", "CoolingDegree", "Cooling_Drop_24h", "实际电量", "预测用户用电量", "预测误差"]].copy()
    res["Day_Type"] = res["Day_Type"].map({0: "工作日", 1: "常规周末", 2: "法定节假日"})
    res["偏差率%"] = (res["预测用户用电量"] - res["实际电量"]) / res["实际电量"] * 100
    res.rename(columns={"Day_Type": "日期类型", "Effective_Load": "全网负荷", "CoolingDegree": "降温度数", "Cooling_Drop_24h": "降温幅度"}, inplace=True)

    return res, model, df, feature_cols, K_expansion


# ==============================================================================
# 【Streamlit界面】
# ==============================================================================
st.sidebar.title("⚡ 参数配置")
st.sidebar.subheader("📅 日期设置")
system_start = st.sidebar.date_input("训练起始日期", datetime.date(2026, 1, 1))
run_date = st.sidebar.date_input("运行日期", datetime.date(2026, 8, 7))
target_start = st.sidebar.date_input("预测起始日期", datetime.date(2026, 8, 10))
target_end = st.sidebar.date_input("预测结束日期", datetime.date(2026, 8, 16))

st.sidebar.subheader("📂 数据文件")
uploaded_file = st.sidebar.file_uploader("上传Excel文件", type=["xlsx"])
run_button = st.sidebar.button("🚀 运行预测", type="primary", use_container_width=True)

st.title("⚡ 电力交易用电量预测平台")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📊 基础数据", "📈 预测结果", "📊 可视化", "🎯 模型性能"])

with tab1:
    st.subheader("用户扩容数据")
    expansion_df = pd.DataFrame(USER_EXPANSION).T
    st.dataframe(expansion_df, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(x=list(MONTHLY_ELECTRICITY_CURVE.keys()), y=list(MONTHLY_ELECTRICITY_CURVE.values()))
        fig.update_layout(title="月度用电曲线", height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=list(range(1,25)), y=JULY_SHAPE_WEEKDAY, name='工作日'))
        fig2.add_trace(go.Scatter(x=list(range(1,25)), y=JULY_SHAPE_WEEKEND, name='周末'))
        fig2.update_layout(title="夏季分时形状", height=350)
        st.plotly_chart(fig2, use_container_width=True)

if run_button and uploaded_file is not None:
    with st.spinner("正在执行预测..."):
        df_input = pd.read_excel(uploaded_file)
        result, model, df_full, feature_cols, K = run_master_forecast(
            df_input, str(system_start), str(run_date), str(target_start), str(target_end)
        )
        st.session_state['result'] = result
        st.session_state['model'] = model
        st.session_state['feature_cols'] = feature_cols
        st.session_state['K'] = K
        st.success("✅ 预测完成！")

if 'result' in st.session_state:
    result = st.session_state['result']
    model = st.session_state['model']
    feature_cols = st.session_state['feature_cols']
    K = st.session_state['K']

    with tab2:
        st.subheader("预测结果")
        eval_mask = result["实际电量"].notna()
        if eval_mask.sum() > 0:
            mape = np.mean(np.abs(result.loc[eval_mask, "偏差率%"]))
            st.metric("整体MAPE", f"{mape:.2f}%")
            st.metric("K_expansion", f"{K:.6f}")
        st.dataframe(result, use_container_width=True)

    with tab3:
        st.subheader("预测vs实际对比")
        fig3 = go.Figure()
        for date in result['Date'].unique()[:3]:
            day = result[result['Date'] == date]
            fig3.add_trace(go.Scatter(x=day['Hour'], y=day['预测用户用电量'], name=f'{date}预测'))
            if day['实际电量'].notna().sum() > 0:
                fig3.add_trace(go.Scatter(x=day['Hour'], y=day['实际电量'], name=f'{date}实际', line=dict(dash='dash')))
        fig3.update_layout(height=500)
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("每日偏差率")
        daily_mape = result[result["实际电量"].notna()].groupby("Date").apply(lambda x: np.mean(np.abs(x["偏差率%"])))
        fig4 = px.bar(x=[str(d) for d in daily_mape.index], y=daily_mape.values)
        fig4.update_layout(xaxis_title="日期", yaxis_title="MAPE%", height=350)
        st.plotly_chart(fig4, use_container_width=True)

    with tab4:
        st.subheader("特征重要性 Top 20")
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        fig5 = px.bar(importance.head(20), x='importance', y='feature', orientation='h')
        fig5.update_layout(height=500)
        st.plotly_chart(fig5, use_container_width=True)
