file_path = "data_csv/product_data.csv"
with open(file_path, "r", encoding="utf-8") as f:
    header = f.readline()
    expected_cols = header.count(",") + 1
    for i, line in enumerate(f, start=2):
        if line.count(",") + 1 != expected_cols:
            print(f"Line {i} has {line.count(',') + 1} columns (expected {expected_cols}):")
            print(line.strip()) 