# Cell 2 — Load Dataset (Notebook cell index: 4)

This cell reads training and test CSVs and renames columns to `text` and `label`.

---

## Granular line-by-line breakdown

### Path constants

```python
TRAIN_PATH = 'Training_data_9.csv'
TEST_PATH = 'Test_data.csv'
```

- **`TRAIN_PATH`** — Variable holding the **filename** (not used later in this exact notebook because absolute paths are hard-coded below). **Why:** Documents expected file names for readers or future refactors.
- **`TEST_PATH`** — Same for test file name.

**If you removed these lines:** No runtime change unless other code referenced them—here they are informational.

---

### Read CSV files

```python
train_df = pd.read_csv("I:/CSE440-Project/Training_data_9.csv")
```

- **`pd`** — Pandas alias from imports cell.
- **`read_csv(path)`** — Reads a comma-separated file into a **DataFrame** (rows = samples, columns = CSV headers).
- **`"I:/CSE440-Project/..."`** — **String literal:** absolute path on the author’s machine. **You must change this** to where your CSV lives.
- **`train_df`** — Variable name holding the training DataFrame.

```python
test_df = pd.read_csv("I:/CSE440-Project/Test_data.csv")
```

- Same pattern for the held-out test set **`test_df`**.

**If path wrong:** `FileNotFoundError`. **If CSV malformed:** pandas may error or parse incorrectly.

---

### Rename columns

```python
train_df = train_df.rename(columns={'News Headline': 'text', 'News Topic': 'label'})
```

- **`train_df.rename(...)`** — Returns a **new** DataFrame with columns renamed (assignment replaces `train_df` variable).
- **`columns={ old: new, ... }`** — Dictionary: **key** = existing header name in CSV, **value** = desired short name.
  - **`'News Headline'` → `'text'`** — Headline string becomes column `text`.
  - **`'News Topic'` → `'label'`** — Topic/class becomes column `label`.

```python
test_df = test_df.rename(columns={'News Headline': 'text', 'News Topic': 'label'})
```

- Same mapping for test so **both** tables share schema.

**If omitted:** Code like `train_df['text']` later raises **`KeyError`** because columns keep original names.

---

### Diagnostics

```python
print('Train shape:', train_df.shape)
```

- **`print`** — Writes to notebook output.
- **`'Train shape:'`** — Literal label string (first argument).
- **`train_df.shape`** — **Attribute:** tuple `(num_rows, num_columns)`.

```python
print('Test shape:', test_df.shape)
```

- Same for test DataFrame.

```python
print('\nLabel distribution in train:')
```

- **`'\n'`** — Newline before “Label distribution…” for readability.

```python
print(train_df['label'].value_counts())
```

- **`train_df['label']`** — Series = one label per row.
- **`.value_counts()`** — Counts rows per distinct label (sorted by default descending).

```python
train_df.head()
```

- **`.head()`** — Method with default **n=5**: returns first 5 rows DataFrame.
- **Last expression in cell** — In Jupyter, displays HTML table of those rows.

---

## Mechanism of the whole cell

Load two CSVs → rename four headers to two canonical names → print shapes and label counts → show sample rows.

---

## Role and importance in the notebook

Defines **`train_df`** and **`test_df`** with **`text`** and **`label`** for every downstream step.
