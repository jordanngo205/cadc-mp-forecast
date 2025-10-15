import math
import os
import re
from datetime import datetime, timedelta
from typing import List

import pandas as pd
import streamlit as st

# ============================= PAGE SETUP =============================
st.set_page_config(page_title="CADC MP Forecast Tool", page_icon="📦", layout="centered")

st.markdown(
    """
    <style>
        div[data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top, #f3f0ff 0%, #ffffff 55%, #f7f9ff 100%);
            padding: 2rem 0 3rem;
        }

        .hero-card {
            background: rgba(255, 255, 255, 0.88);
            border-radius: 20px;
            padding: 26px 34px;
            box-shadow: 0 14px 35px rgba(122, 104, 210, 0.15);
            border: 1px solid rgba(135, 115, 210, 0.25);
            margin-bottom: 2rem;
        }

        .hero-card h1 {
            font-size: 2.1rem;
            margin: 0;
            color: #2a1d52;
        }

        .section-box {
            background: rgba(255, 255, 255, 0.94);
            border-radius: 18px;
            padding: 20px 24px;
            box-shadow: 0 10px 25px rgba(135, 115, 210, 0.10);
            border: 1px solid rgba(153, 128, 226, 0.20);
            margin-bottom: 1.5rem;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #3a296f;
            margin-bottom: 1rem;
        }

        div.stButton > button {
            border-radius: 12px;
            border: none;
            background: linear-gradient(135deg, #755dd8, #f18fb1);
            color: white;
            font-weight: 600;
            letter-spacing: 0.04em;
            padding: 0.6rem 1.2rem;
            box-shadow: 0 10px 24px rgba(121, 87, 216, 0.25);
            transition: all 0.2s ease-in-out;
        }

        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 16px 32px rgba(121, 87, 216, 0.35);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <h1>📦 CADC Manpower Forecast Tool</h1>
        <p>📅 Week-over-week carryover, holiday handling, and 3-3-0 SLA distribution</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================= CONSTANTS =============================
ORDERS_PER_HOUR = 15
HOURS_PER_SHIFT = 8
PICKING_RATE = 95
PICKS_PER_ORDER = 4
RUNNER_RATIO = 5
DATA_FILE = "forecast_history.csv"
CARRY_FILE = "carryover_log.csv"

# ============================= HELPERS =============================
def clean_column_name(name: str) -> str:
    cleaned = re.sub(r"[\r\n\t]+", " ", str(name))
    cleaned = re.sub(r"[^0-9a-zA-Z]+", " ", cleaned)
    return cleaned.strip().lower()

def parse_numeric(values, default=0, dtype=int):
    if isinstance(values, pd.Series):
        series = values
    else:
        series = pd.Series(values)
    cleaned = series.astype(str).str.replace(r"[^\d\.-]", "", regex=True).replace("", pd.NA)
    numeric = pd.to_numeric(cleaned, errors="coerce").fillna(default)
    return numeric.astype(dtype)

def day_name(dt: datetime.date) -> str:
    return dt.strftime("%A")

def is_working_day(name: str, is_peak: bool, is_holiday: bool) -> bool:
    if is_holiday:
        return False
    if is_peak:
        return name != "Sunday"
    return name not in ("Saturday", "Sunday")

def next_working_index(start: int, names: List[str], peaks: List[bool], holidays: List[bool]):
    for j in range(start + 1, len(names)):
        if is_working_day(names[j], peaks[j], holidays[j]):
            return j
    return None

# ============================= INPUT SECTION =============================
st.markdown('<div class="section-box"><div class="section-title">⚙️ Weekly Inputs</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)
with col_a:
    fte_present = st.number_input("FTE Present", min_value=0, value=10)
with col_b:
    ft_off = st.number_input("FT Off", min_value=0, value=2)
with col_c:
    retail_ttb = st.number_input("Retail/TTB/OTC/Salon", min_value=0, value=9)

week_end = st.date_input("📅 Week End Date (Saturday)")
if week_end.weekday() != 5:
    st.warning("⚠️ Expected a Saturday week end. The tool assumes the previous Sunday as the start.")
week_start = week_end - timedelta(days=6)
week_dates = [week_start + timedelta(days=i) for i in range(7)]
prev_saturday = week_start - timedelta(days=1)

carry_default = 0
if os.path.exists(CARRY_FILE):
    carry_log = pd.read_csv(CARRY_FILE)
    col = "SaturdayOrders" if "SaturdayOrders" in carry_log.columns else "CarryOverOut" if "CarryOverOut" in carry_log.columns else None
    if col and not carry_log.empty:
        carry_default = float(carry_log[col].iloc[-1])
        st.info(f"📦 Carryover from previous Saturday: **{carry_default:,.0f} orders**")

prev_sat_orders = st.number_input(
    f"📦 Previous Saturday ({prev_saturday.strftime('%a %b %d, %Y')}) Orders",
    min_value=0,
    value=int(carry_default),
    step=1,
)

st.markdown("#### 🧾 Daily Orders & Flags")
orders, peaks, holidays = [], [], []
for dt in week_dates:
    label = dt.strftime("%a %b %d, %Y")
    c1, c2, c3 = st.columns([1.5, 0.6, 0.6])
    val = c1.number_input(label, min_value=0, step=1, value=0, key=f"orders_{dt}")
    peak_flag = c2.checkbox("Peak", key=f"peak_{dt}")
    holiday_flag = c3.checkbox("Holiday", key=f"holiday_{dt}")
    orders.append(val)
    peaks.append(peak_flag)
    holidays.append(holiday_flag)

st.markdown("</div>", unsafe_allow_html=True)

# ============================= CALCULATIONS =============================
day_names = [day_name(dt) for dt in week_dates]
shifts = [3 if peaks[i] else 2 for i in range(len(week_dates))]

Day1 = [o * 0.5 for o in orders]
Day2 = [o * 0.5 for o in orders]

FriAdj = [0.0] * 7
MonAdj = [0.0] * 7
TueAdj = [0.0] * 7

for i, name in enumerate(day_names):
    if name == "Friday" and i >= 1:
        FriAdj[i] = orders[i] + Day2[i - 1]
    elif name == "Monday":
        sunday_idx = i - 1
        mon_adj = prev_sat_orders
        if sunday_idx >= 0:
            mon_adj += Day1[sunday_idx]
        MonAdj[i] = mon_adj + Day1[i]
    elif name == "Tuesday":
        sunday_idx = i - 2
        monday_idx = i - 1
        TueAdj[i] = (
            (Day2[sunday_idx] if sunday_idx >= 0 else 0.0)
            + (Day2[monday_idx] if monday_idx >= 0 else 0.0)
            + Day1[i]
        )

NormalSLA, PeakSLA = [], []
for i, name in enumerate(day_names):
    PeakSLA.append(Day1[i] + (Day2[i - 1] if i > 0 and peaks[i] else 0.0))
    if name in ("Saturday", "Sunday"):
        normal_val = 0.0
    elif name == "Monday":
        normal_val = MonAdj[i]
    elif name == "Tuesday":
        normal_val = TueAdj[i]
    elif name == "Friday":
        normal_val = FriAdj[i]
    else:
        normal_val = Day1[i] + (Day2[i - 1] if i > 0 else 0.0)
    NormalSLA.append(normal_val)

base_sla = [PeakSLA[i] if peaks[i] else NormalSLA[i] for i in range(7)]
processed = base_sla[:]
for i in range(7):
    if holidays[i] and processed[i] > 0:
        carry = processed[i]
        processed[i] = 0.0
        nxt = next_working_index(i, day_names, peaks, holidays)
        if nxt is not None:
            processed[nxt] += carry

process_series = pd.Series(processed)
pack_hc, pick_hc, total_hc, mp_required = [], [], [], []
for i, orders_today in enumerate(process_series):
    if orders_today <= 0:
        pack_hc.append(0.0)
        pick_hc.append(0.0)
        total_hc.append(0)
        mp_required.append(0)
        continue

    shift_count = shifts[i]
    pack = orders_today / (ORDERS_PER_HOUR * HOURS_PER_SHIFT * shift_count)
    pick = (orders_today / PICKS_PER_ORDER) / (PICKING_RATE * HOURS_PER_SHIFT * shift_count)
    runner = pack / RUNNER_RATIO
    total = math.ceil(pack + pick + runner)

    pack_hc.append(round(pack, 2))
    pick_hc.append(round(pick, 2))
    total_hc.append(total)
    mp = total + retail_ttb - fte_present + ft_off
    mp_required.append(max(0, math.ceil(mp)))

df = pd.DataFrame({
    "Date": week_dates,
    "Day": day_names,
    "IsPeak": peaks,
    "IsHoliday": holidays,
    "Orders": orders,
    "Day1": Day1,
    "Day2": Day2,
    "NormalSLA": NormalSLA,
    "PeakSLA": PeakSLA,
    "Processed Orders": processed,
    "PackingHC": pack_hc,
    "PickingHC": pick_hc,
    "TotalHC": total_hc,
    "MPRequired": mp_required,
})

# ============================= DISPLAY =============================
st.markdown('<div class="section-box"><div class="section-title">📊 Daily Breakdown</div>', unsafe_allow_html=True)
st.dataframe(df, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-box"><div class="section-title">📈 Weekly Totals</div>', unsafe_allow_html=True)
total_processed_week = df["Processed Orders"].sum()
total_hc_week = df["TotalHC"].sum()
total_mp_week = df["MPRequired"].sum()
c1, c2, c3 = st.columns(3)
c1.metric("✅ Processed Orders", f"{total_processed_week:,.0f}")
c2.metric("👷 Total HC", f"{total_hc_week:,.0f}")
c3.metric("🧑‍🔧 Total MP Required", f"{total_mp_week:,.0f}")
st.markdown("</div>", unsafe_allow_html=True)

# ============================= SAVE HISTORY =============================
st.markdown('<div class="section-box"><div class="section-title">💾 Save Forecast</div>', unsafe_allow_html=True)

# --- Save Weekly Forecast ---
if st.button("💾 Save Weekly Forecast"):
    week_end_date = week_end.strftime("%Y-%m-%d")
    df_to_store = df.copy()
    df_to_store["WeekEnd"] = week_end_date

    week_exists = False

    # ====== HANDLE FORECAST HISTORY ======
    if os.path.exists(DATA_FILE):
        hist = pd.read_csv(DATA_FILE)
        if "WeekEnd" in hist.columns and week_end_date in hist["WeekEnd"].values:
            week_exists = True
            hist = hist[hist["WeekEnd"] != week_end_date]  # remove old week first
        hist = pd.concat([hist, df_to_store], ignore_index=True)
    else:
        hist = df_to_store

    hist.to_csv(DATA_FILE, index=False)

    # ====== HANDLE CARRYOVER ======
    saturday_orders = float(df[df["Day"] == "Saturday"]["Orders"].iloc[0])
    carry_entry = pd.DataFrame({"Week_End_Date": [week_end_date], "SaturdayOrders": [saturday_orders]})

    if os.path.exists(CARRY_FILE):
        carry_hist = pd.read_csv(CARRY_FILE)
        if "Week_End_Date" in carry_hist.columns and week_end_date in carry_hist["Week_End_Date"].values:
            carry_hist = carry_hist[carry_hist["Week_End_Date"] != week_end_date]  # overwrite
        carry_hist = pd.concat([carry_hist, carry_entry], ignore_index=True)
    else:
        carry_hist = carry_entry

    carry_hist.to_csv(CARRY_FILE, index=False)

    # ====== SUCCESS BANNER ======
    if week_exists:
        st.success(f"✅ Week overwritten successfully for **{week_end_date}**", icon="✅")
    else:
        st.success(f"✅ Week added successfully for **{week_end_date}**", icon="✅")

# --- Reset Tables ---
st.markdown("---")
if st.button("🧹 Reset Forecast & Carryover Tables"):
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    if os.path.exists(CARRY_FILE):
        os.remove(CARRY_FILE)
    st.toast("🧾 All forecast and carryover data cleared.", icon="✅")
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)


# ============================= HISTORY =============================
st.divider()
st.markdown('<div class="section-box"><div class="section-title">📚 Forecast History</div>', unsafe_allow_html=True)
if os.path.exists(DATA_FILE):
    hist = pd.read_csv(DATA_FILE)
    st.dataframe(hist.tail(20), use_container_width=True)
else:
    st.info("ℹ️ No forecast history yet.")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-box"><div class="section-title">📦 Carryover Summary</div>', unsafe_allow_html=True)
if os.path.exists(CARRY_FILE):
    carry_hist = pd.read_csv(CARRY_FILE)
    st.dataframe(carry_hist, use_container_width=True)
    carry_col = 'SaturdayOrders' if 'SaturdayOrders' in carry_hist.columns else 'CarryOverOut'
    if carry_col in carry_hist:
        st.caption(f"📊 Total Saturday orders logged: **{carry_hist[carry_col].sum():,.0f}**")
else:
    st.info("ℹ️ No carryover history yet.")
st.markdown("</div>", unsafe_allow_html=True)
