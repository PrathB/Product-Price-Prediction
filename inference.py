import re
import torch
import joblib
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

test_df = pd.read_csv("sample_test(1).csv")
test_df = test_df.drop(['image_link'],axis='columns')
test_df['catalog_content'] = test_df['catalog_content'].str.lower()

test_df['quantity_value'] = test_df['catalog_content'].str.extract(r'value:\s*([^\n]+)')
test_df['quantity_unit'] = test_df['catalog_content'].str.extract(r'unit:\s*([^\n]+)')

standard_units = {
    "ounce": ["oz", "ounce", "ounces", "oz."],
    "fluid ounce": ["fl oz", "fl. oz", "fl.oz", "fluid ounce", "fluid ounces", "fluid ounce(s)", "fl ounce", "fl. oz."],
    "gram": ["gram", "grams", "gramm", "gr", "grams(gm)"],
    "kg": ["kg","kilogram","kgs","kilograms"],
    "pound": ["pound", "pounds", "lb", "lbs"],
    "ml": ["ml", "millilitre", "milliliter"],
    "litre": ["litre", "liters", "ltr"],
    "count": ["count", "ct", "each", "piece","pc", "unit", "units"],
}
unit_map = {}
for standard, variants in standard_units.items():
    for v in variants:
        unit_map[v.strip().lower()] = standard

test_df['quantity_unit_clean'] = (
    test_df['quantity_unit']
    .astype(str)
    .str.strip()
    .str.lower()
    .map(unit_map)  # map to standard unit
    .fillna('other')  # everything else as 'other'
)

# Pre-build regex dynamically
unit_pattern = '|'.join(map(re.escape, unit_map.keys()))
base_pattern = rf'(\d+(?:\.\d+)?)\s*(?:{unit_pattern})'
pack_pattern = r'(?:pack of\s*|x\s*)(\d+)'

def extract_quantity_and_unit(text):
    if not isinstance(text, str):
        return None, None
    text = text.lower()

    # Base unit extraction
    base_match = re.search(base_pattern, text)
    pack_match = re.search(pack_pattern, text)

    total_qty, unit_clean = None, None
    if base_match:
        qty = float(base_match.group(1))
        raw_unit = base_match.group(0).replace(str(qty), '').strip()
        # clean raw unit
        for u in unit_map:
            if u in raw_unit:
                unit_clean = unit_map[u]
                break

        # Pack multiplier
        if pack_match:
            pack_qty = int(pack_match.group(1))
            total_qty = qty * pack_qty
        else:
            total_qty = qty

    elif pack_match:
        pack_qty = int(pack_match.group(1))
        total_qty = pack_qty
        unit_clean = "count"

    return total_qty, unit_clean

mask = test_df['quantity_unit_clean'] == 'other'

def update_row(row):
    if row['quantity_unit_clean'] == 'other':
        qty, unit = extract_quantity_and_unit(row['catalog_content'])
        if qty is not None and unit is not None:
            row['quantity_value'] = qty
            row['quantity_unit_clean'] = unit
        else:
          row['quantity_value'] = 1
    return row

test_df = test_df.apply(update_row, axis=1)

test_df.loc[test_df['quantity_unit_clean'] == 'other', 'quantity_value'] = 1
test_df.loc[test_df['quantity_unit_clean'] == 'other', 'quantity_unit_clean'] = "count"
print(test_df['quantity_unit_clean'].value_counts())

weight_conversion = {
    'gram': 1,
    'kg': 1000,
    'pound': 453.592,
    'ounce': 28.3495
}

volume_conversion = {
    'ml': 1,
    'litre': 1000,
    'fluid ounce': 29.5735
}

def convert_value_to_standard(row):
    unit = row['quantity_unit_clean']
    qty = row['quantity_value']

    try:
        qty = float(qty)
    except (ValueError, TypeError):
        return 1

    if unit in weight_conversion:
        return round(qty * weight_conversion[unit],2)
    elif unit in volume_conversion:
        return round(qty * volume_conversion[unit],2)
    else:
         return int(qty)

def convert_unit_to_standard(row):
    unit = row['quantity_unit_clean']

    if unit in weight_conversion:
        return "gram"
    elif unit in volume_conversion:
        return "ml"
    else:
        return unit

test_df['quantity_value_standard'] = test_df.apply(convert_value_to_standard, axis=1)
test_df['quantity_unit_standard'] = test_df.apply(convert_unit_to_standard,axis=1)

test_df.head()

import joblib

scaler = joblib.load("scaler.pkl")
test_df['quantity_value_scaled'] = scaler.transform(test_df[['quantity_value_standard']])
test_df.head()

test_df['catalog_content'] = test_df['catalog_content'].str.lower().str.strip()

tokenizer = BertTokenizer.from_pretrained("fine_tuned_bert_with_quantity")

test_encodings = tokenizer(
    test_df['catalog_content'].tolist(),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors='pt'
)

class PriceDataset(Dataset):
    def __init__(self, df, encodings):
        self.df = df
        self.encodings = encodings

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'quantity': torch.tensor(self.df.iloc[idx]['quantity_value_scaled'], dtype=torch.float)
        }
        return item

class BertWithQuantity(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(BertWithQuantity, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(self.bert.config.hidden_size + 1, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask, quantity_value):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        combined = torch.cat((cls_output, quantity_value.unsqueeze(1)), dim=1)
        x = self.dropout(combined)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = BertWithQuantity(model_name="fine_tuned_bert_with_quantity")
model.load_state_dict(torch.load("fine_tuned_bert_with_quantity/pytorch_model.bin", map_location=device))
model.to(device)
model.eval()

dataset = PriceDataset(test_df, test_encodings)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

preds = []

with torch.no_grad():
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        quantity_value = batch['quantity'].to(device)

        outputs = model(input_ids, attention_mask, quantity_value)
        preds.extend(outputs.cpu().numpy())

# Reverse log1p
test_df['predicted_price'] = np.expm1(preds)

test_df.to_csv("predicted_output.csv", index=False)
print("Saved predictions to predicted_output.csv")

