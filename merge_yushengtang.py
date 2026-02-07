import pandas as pd
from pathlib import Path
from decimal import Decimal, InvalidOperation


def _normalize_assessment(value) -> str:
    """Normalize assessment number to plain string, avoiding scientific notation."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        dec = Decimal(text)
        formatted = format(dec, 'f')
        if '.' in formatted:
            formatted = formatted.rstrip('0').rstrip('.') or '0'
        return formatted
    except InvalidOperation:
        return text


def load_xlsx(excel_path: Path) -> pd.DataFrame:
    """Load Excel and ensure assessment id is string (no scientific notation)."""
    df = pd.read_excel(
        excel_path,
        dtype=str,  # read as string first
        keep_default_na=False,
        converters={0: _normalize_assessment},  # first column is id
    )
    if df.empty:
        raise ValueError("Excel is empty: {}".format(excel_path))
    id_col = df.columns[0]
    df['AssessmentNumber'] = df[id_col].apply(_normalize_assessment)
    return df


def load_tongue_csv(csv_path: Path) -> pd.DataFrame:
    """Load tongue CSV, normalize assessment id column to 'AssessmentNumber'."""
    df = pd.read_csv(csv_path, dtype=str)
    # If index was saved without name, pandas will give it an unnamed column
    if 'AssessmentNumber' not in df.columns:
        if 'Unnamed: 0' in df.columns:
            df = df.rename(columns={'Unnamed: 0': 'AssessmentNumber'})
        else:
            # Fall back to first column
            first_col = df.columns[0]
            df = df.rename(columns={first_col: 'AssessmentNumber'})
    df['AssessmentNumber'] = df['AssessmentNumber'].astype(str)
    return df


def merge_yushengtang():
    base = Path('datasets')
    excel_path = base / 'yushengtang.xlsx'
    tongue_csv_path = base / 'yushengtang_dataTongue.csv'
    output_csv_path = base / 'yushengtang.csv'

    df_excel = load_xlsx(excel_path)
    df_tongue = load_tongue_csv(tongue_csv_path)

    merged = df_excel.merge(df_tongue, on='AssessmentNumber', how='left', suffixes=('', '_tongue'))
    merged = merged.set_index('AssessmentNumber')

    # 保持字符串写出，避免科学计数法
    merged.to_csv(output_csv_path, encoding='utf-8-sig')
    print(f"Merged rows: {len(merged)} -> {output_csv_path}")


if __name__ == '__main__':
    merge_yushengtang()
