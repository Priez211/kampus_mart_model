import csv

input_path = "data_csv/product_data.csv"
output_path = "data_csv/product_data_cleaned.csv"

with open(input_path, newline='', encoding='utf-8') as infile, open(output_path, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    for row in reader:
        # Remove commas from price fields (assume columns 4 and 5, index 4 and 5)
        if len(row) > 5:
            row[4] = row[4].replace(',', '')
            row[5] = row[5].replace(',', '')
        writer.writerow(row)

print(f"Cleaned file saved as {output_path}") 