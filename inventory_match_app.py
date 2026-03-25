from flask import Flask, request, jsonify
import pandas as pd
import re
from rapidfuzz import process, fuzz
from unidecode import unidecode
import os

app = Flask(__name__)

# =====================================
# NORMALIZATION
# =====================================
def normalize_text(text):
    if pd.isna(text) or not text:
        return ""
    text = unidecode(str(text))
    text = text.lower()
    # Remove common noise words for inventory descriptions
    remove_words = [
        "the", "with", "and", "for", "per", "set", "of", "a", "an",
        "mm", "cm", "inch", "in", "pack", "pcs", "piece", "pieces",
        "unit", "units", "ea", "each"
    ]
    pattern = r'\b(' + '|'.join(remove_words) + r')\b'
    text = re.sub(pattern, '', text)
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_code(code):
    """
    Normalise an alt code for tolerant matching:
    - Lowercase
    - Remove underscores (customers often omit them)
    - Collapse O/0 ambiguity by mapping both to '0'
    """
    if pd.isna(code) or not code:
        return ""
    code = str(code).lower()
    code = code.replace('_', '')       # ignore underscore presence/absence
    code = code.replace('o', '0')     # treat letter O and digit 0 as the same
    return code

# =====================================
# LOAD INVENTORY DATABASE ON STARTUP
# =====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_df = pd.read_excel(os.path.join(BASE_DIR, 'Inventory_Items_-_active_.xlsx'))
db_df.columns = db_df.columns.str.strip()

# Normalise key fields
db_df['Inventory ID'] = db_df['Inventory ID'].fillna('').astype(str).str.strip()
db_df['Inventory ID.1'] = db_df['Inventory ID.1'].fillna('').astype(str).str.strip()
db_df['Description'] = db_df['Description'].fillna('').astype(str).str.strip()
db_df['Barcode'] = db_df['Barcode'].fillna('').astype(str).str.strip().str.split('.').str[0]  # remove .0 suffix

db_df['normalized_description'] = db_df['Description'].apply(normalize_text)
db_df['normalized_alt_code'] = db_df['Inventory ID.1'].apply(normalize_text)
db_df['normalized_code'] = db_df['Inventory ID.1'].apply(normalize_code)  # underscore/O0 tolerant

# =====================================
# GET TOP 5 CANDIDATES BY DESCRIPTION
# =====================================
def get_top5(query, n=5):
    norm_query = normalize_text(query)
    if not norm_query:
        return []
    db_list = db_df['normalized_description'].tolist()
    matches = process.extract(
        norm_query,
        db_list,
        scorer=fuzz.token_sort_ratio,
        limit=n
    )
    results = []
    for match, score, idx in matches:
        row = db_df.iloc[idx]
        results.append({
            'inventory_id': str(row.get('Inventory ID', '')),
            'alt_code': str(row.get('Inventory ID.1', '')),
            'description': str(row.get('Description', '')),
            'item_class': str(row.get('Item Class', '')),
            'item_class_description': str(row.get('Item Class Description', '')),
            'default_warehouse': str(row.get('Default Warehouse', '')),
            'barcode': str(row.get('Barcode', '')),
            'score': score
        })
    return results

def build_result(row, score, match_method, top5):
    return {
        'inventory_id': str(row.get('Inventory ID', '')),
        'alt_code': str(row.get('Inventory ID.1', '')),
        'description': str(row.get('Description', '')),
        'item_class': str(row.get('Item Class', '')),
        'item_class_description': str(row.get('Item Class Description', '')),
        'default_warehouse': str(row.get('Default Warehouse', '')),
        'barcode': str(row.get('Barcode', '')),
        'confidence_score': score,
        'match_method': match_method,
        'needs_review': score < 80,
        'top5_candidates': top5
    }

# =====================================
# MATCH ENDPOINT
# =====================================
@app.route('/match_inventory', methods=['POST'])
def match_inventory():
    data = request.json
    inventory_id = str(data.get('inventoryId', '') or '').strip()
    alt_code = str(data.get('altCode', '') or '').strip()
    barcode = str(data.get('barcode', '') or '').strip().split('.')[0]  # handle float barcodes
    description = str(data.get('description', '') or '').strip()

    best_match = None
    best_score = 0
    match_method = None

    # 1. Exact Inventory ID match (strongest)
    if inventory_id:
        exact = db_df[db_df['Inventory ID'].str.lower() == inventory_id.lower()]
        if not exact.empty:
            best_match = exact.iloc[0]
            best_score = 100
            match_method = 'inventory_id_exact'

    # 2. Exact Alt Code (Inventory ID.1) match
    if best_match is None and alt_code:
        exact = db_df[db_df['Inventory ID.1'].str.lower() == alt_code.lower()]
        if not exact.empty:
            best_match = exact.iloc[0]
            best_score = 100
            match_method = 'alt_code_exact'

    # 2b. Tolerant Alt Code match — ignores underscores and O/0 confusion
    if best_match is None and alt_code:
        norm_input = normalize_code(alt_code)
        if norm_input:
            tolerant = db_df[db_df['normalized_code'] == norm_input]
            if not tolerant.empty:
                best_match = tolerant.iloc[0]
                best_score = 98
                match_method = 'alt_code_tolerant'

    # 3. Exact Barcode match
    if best_match is None and barcode:
        exact = db_df[db_df['Barcode'] == barcode]
        if not exact.empty:
            best_match = exact.iloc[0]
            best_score = 100
            match_method = 'barcode_exact'

    # 4. Fuzzy Alt Code match
    if best_match is None and alt_code:
        norm_alt = normalize_text(alt_code)
        if norm_alt:
            result = process.extractOne(
                norm_alt,
                db_df['normalized_alt_code'].tolist(),
                scorer=fuzz.token_sort_ratio
            )
            if result and result[1] > best_score:
                best_match = db_df.iloc[result[2]]
                best_score = result[1]
                match_method = 'alt_code_fuzzy'

    # 5. Fuzzy Description match
    if (best_match is None or best_score < 80) and description:
        result = process.extractOne(
            normalize_text(description),
            db_df['normalized_description'].tolist(),
            scorer=fuzz.token_sort_ratio
        )
        if result and result[1] > best_score:
            best_match = db_df.iloc[result[2]]
            best_score = result[1]
            match_method = 'description_fuzzy'

    # Top 5 candidates for AI fallback
    query = description or alt_code or inventory_id or ''
    top5 = get_top5(query)

    if best_match is not None:
        return jsonify(build_result(best_match, best_score, match_method, top5))
    else:
        return jsonify({
            'inventory_id': None,
            'alt_code': None,
            'description': None,
            'item_class': None,
            'item_class_description': None,
            'default_warehouse': None,
            'barcode': None,
            'confidence_score': 0,
            'match_method': None,
            'needs_review': True,
            'top5_candidates': top5
        })

if __name__ == '__main__':
    app.run(debug=True)
