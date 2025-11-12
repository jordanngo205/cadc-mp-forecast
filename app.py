import io
import math
import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# ============================= PAGE SETUP =============================
st.set_page_config(page_title="CADC MP Forecast Tool", page_icon="📦", layout="centered")

st.markdown(
    """
    <style>
        div[data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top, #f4f2ff 0%, #ffffff 55%, #f7f9ff 100%);
            padding: 2rem 0 3rem;
        }
        .hero-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 24px;
            padding: 30px 40px;
            box-shadow: 0 18px 40px rgba(124, 104, 210, 0.18);
            border: 1px solid rgba(136, 116, 212, 0.25);
            margin-bottom: 2rem;
        }
        .hero-card h1 {
            font-size: 2.2rem;
            margin: 0;
            color: #2a1d52;
        }
        .hero-card p {
            margin: 0.45rem 0 0;
            color: #534884;
        }

        /* Section box */
        .section-box {
            background: rgba(255, 255, 255, 0.96);
            border-radius: 20px;
            padding: 26px 30px;
            box-shadow: 0 16px 36px rgba(135, 115, 210, 0.16);
            border: 1px solid rgba(151, 128, 226, 0.18);
            margin-bottom: 2rem;
        }
        .section-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: #3a296f;
            margin-bottom: 1.2rem;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        /* Daily input row spacing */
        div[data-testid="stHorizontalBlock"] {
            margin-bottom: 0.5rem !important;
        }
        label, input, div[data-baseweb="checkbox"] {
            margin-top: 0.25rem !important;
        }

        /* Column spacing balance */
        [data-testid="stColumn"] {
            padding-right: 0.4rem !important;
            padding-left: 0.4rem !important;
        }

        /* Better mobile responsiveness */
        @media (max-width: 800px) {
            .section-box {
                padding: 18px;
            }
            div[data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
            }
            [data-testid="stColumn"] {
                flex: 1 1 45% !important;
                margin-bottom: 0.4rem;
            }
        }

        /* Buttons */
        div.stButton > button {
            border-radius: 12px;
            border: none;
            background: linear-gradient(135deg, #7a5bdc, #f29bc1);
            color: white;
            font-weight: 600;
            letter-spacing: 0.04em;
            padding: 0.6rem 1.2rem;
            box-shadow: 0 14px 28px rgba(121, 87, 216, 0.28);
            transition: all 0.18s ease-in-out;
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 18px 36px rgba(121, 87, 216, 0.36);
        }

        /* Prevent checkbox labels like "Holiday" from wrapping */
        div[data-baseweb="checkbox"] {
            display: flex !important;
            align-items: center !important;
            white-space: nowrap !important;
        }

        /* Make checkbox labels slightly smaller so both fit side by side */
        div[data-baseweb="checkbox"] label {
            font-size: 0.92rem !important;
            white-space: nowrap !important;
            line-height: 1.1 !important;
            margin-left: 4px !important;
        }

        /* Adjust column spacing so checkboxes stay centered and inline */
        [data-testid="stColumn"] {
            padding-right: 0.4rem !important;
            padding-left: 0.4rem !important;
        }


    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero-card">
        <h1>📦 CADC Manpower Forecast Tool</h1>
        <p>📅 Handles carryover, holiday pushes, and the 3-3-0 SLA across the week.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================= CONSTANTS =============================
ORDERS_PER_HOUR = 15
HOURS_PER_SHIFT = 6
PICKING_RATE = 100
RUNNER_RATIO = 5
DATA_FILE = "forecast_history.csv"
CARRY_FILE = "carryover_log.csv"

DEFAULT_FTE = 10
DEFAULT_FT_OFF = 2
DEFAULT_RETAIL = 9
BRANDS_OF_INTEREST = {"clinique", "estee lauder"}
PREFERRED_SHEET = "CA"

# ============================= HELPERS =============================
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


def load_history_defaults(week_end_date: datetime.date) -> Dict[str, Tuple[int, bool, bool, int, int, int]]:
    defaults: Dict[str, Tuple[int, bool, bool, int, int, int]] = {}
    if not os.path.exists(DATA_FILE):
        return defaults
    try:
        history_df = pd.read_csv(DATA_FILE)
        if "WeekEnd" not in history_df.columns:
            return defaults
        key = week_end_date.strftime("%Y-%m-%d")
        subset = history_df[history_df["WeekEnd"] == key]
        if subset.empty or "Date" not in subset.columns:
            return defaults
        for _, row in subset.iterrows():
            try:
                row_date = pd.to_datetime(row["Date"]).date()
            except Exception:
                continue
            try:
                orders_val = int(float(row.get("Orders", 0)))
            except Exception:
                orders_val = 0
            peak_val = row.get("IsPeak", False)
            if isinstance(peak_val, str):
                peak_val = peak_val.strip().lower() == "true"
            holiday_val = row.get("IsHoliday", False)
            if isinstance(holiday_val, str):
                holiday_val = holiday_val.strip().lower() == "true"
            try:
                fte_val = int(float(row.get("FTE", DEFAULT_FTE)))
            except Exception:
                fte_val = DEFAULT_FTE
            try:
                ft_off_val = int(float(row.get("FTOff", DEFAULT_FT_OFF)))
            except Exception:
                ft_off_val = DEFAULT_FT_OFF
            try:
                retail_val = int(float(row.get("RetailSupport", DEFAULT_RETAIL)))
            except Exception:
                retail_val = DEFAULT_RETAIL
            defaults[str(row_date)] = (
                orders_val,
                bool(peak_val),
                bool(holiday_val),
                fte_val,
                ft_off_val,
                retail_val,
            )
    except Exception:
        defaults = {}
    return defaults


def _clean_order_value(value: Optional[str]) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0
    cleaned = str(value).replace('"', "").replace("'", "").strip()
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("\u00a0", "").replace(" ", "")
    if cleaned in {"", "-", "\u2013"}:
        return 0
    try:
        return int(round(float(cleaned)))
    except ValueError:
        return 0


def _is_excel_file(file_name: str) -> bool:
    lowered = (file_name or "").lower()
    return lowered.endswith((".xlsx", ".xls", ".xlsm", ".xlsb"))


def _normalize_forecast_csv(upload_file) -> pd.DataFrame:
    if upload_file is None:
        return pd.DataFrame(columns=["WeekEnd", "Date", "Brand", "Orders"])

    file_name = getattr(upload_file, "name", "") or ""
    is_excel = _is_excel_file(file_name)

    raw_bytes: Optional[bytes] = None
    try:
        raw_bytes = upload_file.read()
    except Exception:
        raw_bytes = None
    if isinstance(raw_bytes, str):
        raw_bytes = raw_bytes.ezncode("utf-8")

    buffer: io.BytesIO
    if raw_bytes is not None:
        buffer = io.BytesIO(raw_bytes)
    else:
        if hasattr(upload_file, "getvalue"):
            value = upload_file.getvalue()  # type: ignore[attr-defined]
            if isinstance(value, str):
                value = value.encode("utf-8")
            buffer = io.BytesIO(value)
        else:
            data = upload_file.read()  # type: ignore[call-arg]
            if isinstance(data, str):
                data = data.encode("utf-8")
            buffer = io.BytesIO(data)

    try:
        if is_excel:
            buffer.seek(0)
            try:
                excel_file = pd.ExcelFile(buffer)
            except Exception as exc:
                raise ValueError(f"Unable to read Excel workbook: {exc}") from exc
            target_sheet = PREFERRED_SHEET if PREFERRED_SHEET in excel_file.sheet_names else excel_file.sheet_names[0]
            try:
                raw_df = pd.read_excel(
                    excel_file,
                    sheet_name=target_sheet,
                    header=None,
                    dtype=str,
                )
            except Exception as exc:
                raise ValueError(f"Unable to read sheet '{target_sheet}': {exc}") from exc
        else:
            buffer.seek(0)
            try:
                raw_df = pd.read_csv(
                    buffer,
                    header=None,
                    dtype=str,
                    engine="python",
                )
            except Exception as exc:
                raise ValueError(f"Unable to read CSV: {exc}") from exc
    finally:
        try:
            upload_file.seek(0)
        except Exception:
            pass

    def _normalize_cell(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and math.isnan(value):
            return ""
        return str(value).strip()

    header_idx: Optional[int] = None
    for idx, row in raw_df.iterrows():
        normalized_cells = [_normalize_cell(cell).lower() for cell in row.tolist()]
        if (
            len(normalized_cells) >= 4
            and normalized_cells[0] == "week end date"
            and normalized_cells[1] == "date"
            and normalized_cells[2] == "brand"
            and "order" in normalized_cells[3]
        ):
            header_idx = idx
            break

    if header_idx is None:
        raise ValueError("CSV is missing required columns: Week End Date, Date, Brand, Orders.")

    header_values_raw = raw_df.iloc[header_idx].tolist()
    header_values = [_normalize_cell(cell) for cell in header_values_raw]
    data_df = raw_df.iloc[header_idx + 1 :].copy().reset_index(drop=True)

    node_values = (
        [_normalize_cell(cell) for cell in raw_df.iloc[header_idx - 1].tolist()]
        if header_idx and header_idx > 0
        else [""] * len(header_values)
    )

    def find_column_index(target: str) -> int:
        target_lower = target.lower()
        for idx, name in enumerate(header_values):
            if name.lower() == target_lower:
                return idx
        raise ValueError(target)

    try:
        week_idx = find_column_index("Week End Date")
        date_idx = find_column_index("Date")
        brand_idx = find_column_index("Brand")
    except ValueError as err:
        raise ValueError("CSV is missing required columns: Week End Date, Date, Brand, Orders.") from err

    def column_has_data(col_index: int) -> bool:
        column = data_df.iloc[:, col_index]
        if column.dropna().empty:
            return False
        for val in column:
            cell = _normalize_cell(val).lower()
            if cell not in ("", "-", "nan", "none", "0"):
                return True
        return False

    def choose_orders_index(keywords: Tuple[str, ...]) -> Optional[int]:
        for idx, name in enumerate(header_values):
            if "order" not in name.lower():
                continue
            label = node_values[idx].lower() if idx < len(node_values) else ""
            if keywords and not any(key in label for key in keywords):
                continue
            if column_has_data(idx):
                return idx
        return None

    orders_idx: Optional[int] = None
    for keys in (("all",), ("cadc", "elc"), tuple()):
        candidate = choose_orders_index(keys)
        if candidate is not None:
            orders_idx = candidate
            break

    if orders_idx is None:
        raise ValueError("CSV is missing required columns: Week End Date, Date, Brand, Orders.")

    selected = pd.DataFrame({
        "Week End Date": data_df.iloc[:, week_idx],
        "Date": data_df.iloc[:, date_idx],
        "Brand": data_df.iloc[:, brand_idx],
        "Orders": data_df.iloc[:, orders_idx],
    })

    normalized = selected.apply(lambda col: col.map(_normalize_cell))
    normalized = normalized[
        (normalized["Week End Date"] != "")
        | (normalized["Date"] != "")
        | (normalized["Brand"] != "")
        | (normalized["Orders"] != "")
    ]

    normalized = pd.DataFrame({
        "WeekEnd": pd.to_datetime(normalized["Week End Date"], errors="coerce").dt.date,
        "Date": pd.to_datetime(normalized["Date"], errors="coerce").dt.date,
        "Brand": (
            normalized["Brand"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace("\u00e9", "e", regex=False)
            .str.replace("\u00c9", "e", regex=False)
        ),
        "Orders": normalized["Orders"].apply(_clean_order_value),
    })
    return normalized


def parse_forecast_upload(upload_file) -> Tuple[Optional[date], Dict[date, int]]:
    """
    Extract the week end date and combined Clinique + Estee Lauder orders from a forecast CSV.
    """
    normalized = _normalize_forecast_csv(upload_file)
    if normalized.empty:
        return None, {}

    week_values = normalized["WeekEnd"].dropna()
    week_end_date: Optional[date] = week_values.iloc[0] if not week_values.empty else None

    filtered = normalized[normalized["Brand"].isin(BRANDS_OF_INTEREST)].dropna(subset=["Date"])
    if filtered.empty:
        return week_end_date, {}

    orders_by_date = filtered.groupby("Date")["Orders"].sum()
    orders_lookup: Dict[date, int] = {day: int(total) for day, total in orders_by_date.items()}
    return week_end_date, orders_lookup


def parse_bulk_forecast_upload(upload_file) -> Dict[date, Dict[date, int]]:
    """
    Return a mapping of week-end Saturday -> {daily date -> Clinique + Estee Lauder orders}.
    """
    normalized = _normalize_forecast_csv(upload_file)
    if normalized.empty:
        return {}

    filtered = normalized[normalized["Brand"].isin(BRANDS_OF_INTEREST)].dropna(subset=["Date", "WeekEnd"])
    if filtered.empty:
        return {}

    grouped = filtered.groupby(["WeekEnd", "Date"])["Orders"].sum().sort_index()
    results: Dict[date, Dict[date, int]] = {}
    for (week_end_value, day), total in grouped.items():
        week_dict = results.setdefault(week_end_value, {})
        week_dict[day] = int(total)
    return results


# ============================= INPUT SECTION =============================
st.markdown('<div class="section-box"><div class="section-title">⚙️ Weekly Inputs</div>', unsafe_allow_html=True)

uploaded_week_end: Optional[date] = None
uploaded_orders_lookup: Dict[date, int] = {}

tab_weekly, tab_bulk = st.tabs(["Weekly CSV", "Bulk CSV (multi-week)"])

with tab_weekly:
    weekly_file = st.file_uploader(
        "📄 Upload single-week forecast",
        type=["csv", "xlsx", "xls", "xlsm", "xlsb"],
        help="Prefill one week's Clinique + Estee Lauder orders.",
        key="weekly_forecast_upload",
    )
    if weekly_file is not None:
        try:
            detected_week_end, detected_orders = parse_forecast_upload(weekly_file)
            uploaded_week_end = detected_week_end
            uploaded_orders_lookup = detected_orders
            if detected_orders:
                if detected_week_end:
                    st.success(
                        f"Loaded {len(detected_orders)} day(s) for week ending {detected_week_end:%Y-%m-%d}."
                    )
                else:
                    st.success(f"Loaded {len(detected_orders)} day(s) of orders from the CSV.")
            else:
                st.warning("No Clinique or Estee Lauder rows were found in the uploaded CSV.")
            if detected_week_end is None:
                st.info("Week end date was not detected in the file; please choose one manually.")
        except ValueError as err:
            st.error(f"CSV parsing error: {err}")
            uploaded_week_end = None
            uploaded_orders_lookup = {}

with tab_bulk:
    bulk_file = st.file_uploader(
        "📦 Upload multi-week forecast",
        type=["csv", "xlsx", "xls", "xlsm", "xlsb"],
        help="Drop in 6 months (or more) of data and pick the week to prefill.",
        key="bulk_forecast_upload",
    )
    if bulk_file is not None:
        try:
            bulk_orders = parse_bulk_forecast_upload(bulk_file)
            if not bulk_orders:
                st.warning("No Clinique or Estee Lauder rows were found in the bulk CSV.")
            else:
                week_options = sorted(bulk_orders.keys())
                default_week = max(week_options)
                selected_index = week_options.index(default_week)
                selected_week = st.selectbox(
                    "Select week ending Saturday",
                    options=week_options,
                    index=selected_index,
                    format_func=lambda d: d.strftime("%Y-%m-%d"),
                )
                uploaded_week_end = selected_week
                uploaded_orders_lookup = bulk_orders[selected_week]
                st.success(
                    f"Loaded {len(uploaded_orders_lookup)} day(s) from bulk file for week ending {selected_week:%Y-%m-%d}."
                )
        except ValueError as err:
            st.error(f"Bulk CSV parsing error: {err}")
            uploaded_week_end = None
            uploaded_orders_lookup = {}

week_end_default = uploaded_week_end or datetime.today().date()
week_end = st.date_input("📅 Week End Date (Saturday)", value=week_end_default)
if week_end.weekday() != 5:
    st.warning("⚠️ Expected a Saturday week end. The previous Sunday will be used as the start.")
week_start = week_end - timedelta(days=6)
week_dates = [week_start + timedelta(days=i) for i in range(7)]
prev_saturday = week_start - timedelta(days=1)

carry_default = 0
if os.path.exists(CARRY_FILE):
    carry_log = pd.read_csv(CARRY_FILE)
    if not carry_log.empty and "SaturdayOrders" in carry_log.columns:
        carry_default = float(carry_log.iloc[-1]["SaturdayOrders"])
        st.info(f"📦 Carryover from previous Saturday: **{carry_default:,.0f} orders**")

prev_sat_orders = st.number_input(
    f"📦 Previous Saturday ({prev_saturday.strftime('%a %b %d, %Y')}) Orders",
    min_value=0,
    value=int(carry_default),
    step=1,
)

history_defaults = load_history_defaults(week_end)

st.markdown("#### 🧾 Daily Orders & Flags")
orders, peaks, holidays = [], [], []
fte_list, ft_off_list, retail_list = [], [], []
for dt in week_dates:
    label = dt.strftime("%a %b %d, %Y")
    default_orders, default_peak, default_holiday, default_fte, default_ft_off, default_retail = history_defaults.get(
        str(dt),
        (0, False, False, DEFAULT_FTE, DEFAULT_FT_OFF, DEFAULT_RETAIL),
    )
    if dt in uploaded_orders_lookup:
        default_orders = uploaded_orders_lookup[dt]
    c1, c2, c3, c4, c5, c6 = st.columns([1.3, 0.8, 0.8, 0.8, 0.8, 0.9])
    val = c1.number_input(label, min_value=0, step=1, value=int(default_orders), key=f"orders_{week_end}_{dt}")
    peak_flag = c2.checkbox("Peak", value=default_peak, key=f"peak_{week_end}_{dt}")
    holiday_flag = c3.checkbox("Holiday", value=default_holiday, key=f"holiday_{week_end}_{dt}")
    fte_val = c4.number_input("FTE", min_value=0, step=1, value=int(default_fte), key=f"fte_{week_end}_{dt}")
    ft_off_val = c5.number_input("FT off", min_value=0, step=1, value=int(default_ft_off), key=f"ftoff_{week_end}_{dt}")
    retail_val = c6.number_input("Retail", min_value=0, step=1, value=int(default_retail), key=f"retail_{week_end}_{dt}")
    orders.append(val)
    peaks.append(peak_flag)
    holidays.append(holiday_flag)
    fte_list.append(fte_val)
    ft_off_list.append(ft_off_val)
    retail_list.append(retail_val)
st.markdown("</div>", unsafe_allow_html=True)

original_orders = orders.copy()

# ============================= CALCULATIONS =============================
day_names = [day_name(dt) for dt in week_dates]

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

processed = [PeakSLA[i] if peaks[i] else NormalSLA[i] for i in range(7)]
for i in range(7):
    if holidays[i] and processed[i] > 0:
        carry = processed[i]
        processed[i] = 0.0
        nxt = next_working_index(i, day_names, peaks, holidays)
        if nxt is not None:
            processed[nxt] += carry

pack_hc, pick_hc, total_hc, mp_required = [], [], [], []
for i, orders_today in enumerate(processed):
    if orders_today <= 0:
        pack_hc.append(0.0)
        pick_hc.append(0.0)
        total_hc.append(0)
        mp_required.append(0)
        continue
    pack = orders_today / (ORDERS_PER_HOUR * HOURS_PER_SHIFT)
    pick = orders_today / (PICKING_RATE * HOURS_PER_SHIFT)
    runner = pack / RUNNER_RATIO
    total = math.ceil(pack + pick + runner)
    pack_hc.append(round(pack, 2))
    pick_hc.append(round(pick, 2))
    total_hc.append(total)
    mp_required.append(max(0, math.ceil(total + retail_list[i] - fte_list[i] + ft_off_list[i])))


df = pd.DataFrame({
    "Date": week_dates,
    "Day": day_names,
    "IsPeak": peaks,
    "IsHoliday": holidays,
    "Orders": orders,
    "FTE": fte_list,
    "FTOff": ft_off_list,
    "RetailSupport": retail_list,
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

# ============================= DAILY SUMMARY METRICS =============================
st.markdown('<div class="section-box"><div class="section-title">📅 Daily MP Snapshot</div>', unsafe_allow_html=True)
cols = st.columns(3)

# Sort to start from Sunday
day_order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
df_sorted = df.set_index("Day").loc[day_order].reset_index()

for i, (_, row) in enumerate(df_sorted.iterrows()):

    col = cols[i % 3]
    with col:
        st.metric(
            label=f"{row['Day']}\nProcessed: {row['Processed Orders']:,.0f}",
            value=f"MP {int(row['MPRequired'])}",
            delta=f"HC {row['TotalHC']}" if row['TotalHC'] else None,
        )
    if (i % 3) == 2 and i < len(df) - 1:
        cols = st.columns(3)
st.markdown("</div>", unsafe_allow_html=True)

# ============================= DISPLAY =============================
st.markdown('<div class="section-box"><div class="section-title">📊 Daily Breakdown</div>', unsafe_allow_html=True)
st.dataframe(df, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-box"><div class="section-title">📈 Weekly Totals</div>', unsafe_allow_html=True)
total_processed_week = df["Processed Orders"].sum()
total_hc_week = df["TotalHC"].sum()
total_mp_week = df["MPRequired"].sum()
col1, col2, col3 = st.columns(3)
col1.metric("✅ Processed Orders", f"{total_processed_week:,.0f}")
col2.metric("👷 Total HC", f"{total_hc_week:,.0f}")
col3.metric("🧑‍🔧 Total MP Required", f"{total_mp_week:,.0f}")
st.markdown("</div>", unsafe_allow_html=True)

# ============================= SAVE HISTORY =============================
st.markdown('<div class="section-box"><div class="section-title">💾 Save Forecast</div>', unsafe_allow_html=True)

if st.button("💾 Save Weekly Forecast"):
    week_end_str = week_end.strftime("%Y-%m-%d")
    df_to_store = df.copy()
    df_to_store["WeekEnd"] = week_end_str
    week_exists = False

    if os.path.exists(DATA_FILE):
        hist = pd.read_csv(DATA_FILE)
        if "WeekEnd" in hist.columns and week_end_str in hist["WeekEnd"].values:
            week_exists = True
            hist = hist[hist["WeekEnd"] != week_end_str]
        hist = pd.concat([hist, df_to_store], ignore_index=True)
    else:
        hist = df_to_store
    hist.to_csv(DATA_FILE, index=False)

    saturday_orders = float(df[df["Day"] == "Saturday"]["Orders"].iloc[0]) if "Saturday" in df["Day"].values else 0.0
    carry_entry = pd.DataFrame({"Week_End_Date": [week_end_str], "SaturdayOrders": [saturday_orders]})
    if os.path.exists(CARRY_FILE):
        carry_hist = pd.read_csv(CARRY_FILE)
        if "Week_End_Date" in carry_hist.columns and week_end_str in carry_hist["Week_End_Date"].values:
            carry_hist = carry_hist[carry_hist["Week_End_Date"] != week_end_str]
        carry_hist = pd.concat([carry_hist, carry_entry], ignore_index=True)
    else:
        carry_hist = carry_entry
    carry_hist.to_csv(CARRY_FILE, index=False)

    if week_exists:
        st.success(f"✅ Week overwritten successfully for **{week_end_str}**")
    else:
        st.success(f"✅ Week added successfully for **{week_end_str}**")

st.markdown("---")
if st.button("🧹 Reset Forecast & Carryover Tables"):
    for file in [DATA_FILE, CARRY_FILE]:
        if os.path.exists(file):
            os.remove(file)
    st.toast("🧾 All forecast and carryover data cleared.", icon="✅")
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # Streamlit < 1.25 fallback
        st.experimental_rerun()
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
    if "SaturdayOrders" in carry_hist.columns:
        st.caption(f"📊 Total Saturday orders logged: **{carry_hist['SaturdayOrders'].sum():,.0f}**")
else:
    st.info("ℹ️ No carryover history yet.")
st.markdown("</div>", unsafe_allow_html=True)
