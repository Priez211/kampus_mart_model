input_path = "data_csv/product_interactions_data.csv"
output_path = "data_csv/product_interactions_data_fixed.csv"

with open(input_path, "r", encoding="utf-8") as infile:
    lines = infile.readlines()

header = lines[0]
expected_cols = header.count(",") + 1

fixed_lines = [header]
for i, line in enumerate(lines[1:], start=2):
    # Remove trailing whitespace and newlines
    line = line.rstrip("\r\n")
    # Remove trailing commas until the number of columns matches the header
    while line.count(",") + 1 > expected_cols:
        line = line.rstrip(",")
    fixed_lines.append(line + "\n")

with open(output_path, "w", encoding="utf-8") as outfile:
    outfile.writelines(fixed_lines)

print(f"Fixed file saved as {output_path}") 