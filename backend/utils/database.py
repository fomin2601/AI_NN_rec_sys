import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from flask import g, jsonify


def get_database():
    database_file = Path('database.db')
    db = getattr(g, '_database', None)

    if db is None:
        db = g._database = sqlite3.connect('database.db')

    db.row_factory = sqlite3.Row

    return db


def get_item_map(df: pd.DataFrame) -> dict:
    df = pd.read_csv('./files/cosmetic_train.tsv', sep='\t')
    unique_df = df[['item_id', 'name']].drop_duplicates()
    item_map = {}

    for i, row in unique_df.iterrows():
        price = _get_item_lower_price(df, row['item_id'])
        if price is np.NaN:
            continue
        item_map[str(row['item_id'])] = {'name': row['name'], 'price': price}

    return item_map


def _get_item_lower_price(df: pd.DataFrame, item_id: int) -> float:
    item_rows = df.loc[(df.item_id == item_id) & (df.price != 0.0)]
    return item_rows.price.min()
