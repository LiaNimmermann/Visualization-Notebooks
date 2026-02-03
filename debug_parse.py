import re

# Read the LaTeX file
with open(r'c:\Code\Preprocessing-Paper-Code\Visualization-Notebooks\Visualization-Notebooks\Data-Visualization\Sparsity_Citiscapes.tex', 'r') as f:
    content = f.read()

# Extract table rows
rows = content.split('\\\\')
rows = [row.strip() for row in rows if row.strip() and not row.startswith('\\')]

print(f"Total rows: {len(rows)}")
print("\nFirst 5 rows:")
for i, row in enumerate(rows[:5]):
    print(f"\nRow {i}:")
    cols = [col.strip() for col in row.split('&')]
    print(f"  Number of columns: {len(cols)}")
    for j, col in enumerate(cols):
        print(f"  Col {j}: {col[:50]}")
