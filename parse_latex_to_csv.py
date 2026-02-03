import re
import csv


output_path = r'c:\Code\Preprocessing-Paper-Code\Visualization-Notebooks\Visualization-Notebooks\Data-Visualization\Sparsity_Cityscapes_ACDC.csv'

# Read the LaTeX file
with open(r'C:\Code\Preprocessing-Paper-Code\Visualization-Notebooks\Visualization-Notebooks\Data-Visualization\Sparsity_Cityscapes_ADCD.tex', 'r') as f:
    content = f.read()

# Extract table rows
rows = content.split('\\\\')
rows = [row.strip() for row in rows if row.strip()]

# Parse data
data = []
current_arch = None
current_sparsity = None

for row in rows:
    # Skip header and formatting rows
    if any(x in row for x in ['toprule', 'bottomrule', 'Architecture & Preprocessing']):
        continue
    
    # Skip if it doesn't contain metric data
    if '~$\\pm$~' not in row:
        continue
    
    # Split by & to get columns
    cols = [col.strip() for col in row.split('&')]
    
    if len(cols) < 6:
        continue
    
    # Extract and clean up columns
    arch_raw = cols[0]
    preprocessing = cols[1]
    sparsity_raw = cols[2]
    miou_str = cols[3]
    macc_str = cols[4]
    aacc_str = cols[5]
    
    # Clean up architecture
    arch = re.sub(r'\\midrule\s*', '', arch_raw)
    arch = re.sub(r'\\multirow\{.*?\}\{\*\}', '', arch).strip()
    arch = arch.replace('\\', '')
    arch = arch.replace('{', '').replace('}', '')  # Remove braces
    
    # Clean up preprocessing
    preprocessing = preprocessing.replace('\\', '').strip()
    
    # Clean up sparsity
    sparsity = re.sub(r'\\multirow\{.*?\}\{\*\}', '', sparsity_raw).strip()
    sparsity = sparsity.replace('\\', '')
    sparsity = sparsity.replace('_', '_')  # Unescape underscores
    sparsity = sparsity.replace('{', '').replace('}', '')  # Remove braces
    
    # Use current values if empty
    if not arch or arch == '-':
        arch = current_arch
    else:
        current_arch = arch
    
    if not sparsity or sparsity == '-':
        sparsity = current_sparsity if current_sparsity else '-'
    else:
        current_sparsity = sparsity
    
    # Parse value ~$\pm$~ variance format
    def parse_metric(s):
        match = re.search(r'([\d.]+)\s*~\$\\pm\$~\s*([\d.]+)', s)
        if match:
            return float(match.group(1)), float(match.group(2))
        return None, None
    
    miou, miou_var = parse_metric(miou_str)
    macc, macc_var = parse_metric(macc_str)
    aacc, aacc_var = parse_metric(aacc_str)
    
    if miou is not None and arch and preprocessing:
        data.append({
            'Architecture': arch,
            'Preprocessing': preprocessing,
            'Sparsity': sparsity,
            'mIoU': miou,
            'mIoU_var': miou_var,
            'mAcc': macc,
            'mAcc_var': macc_var,
            'aAcc': aacc,
            'aAcc_var': aacc_var
        })

# Write to CSV
with open(output_path, 'w', newline='') as csvfile:
    fieldnames = ['Architecture', 'Preprocessing', 'Sparsity', 'mIoU', 'mIoU_var', 'mAcc', 'mAcc_var', 'aAcc', 'aAcc_var']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

print(f"CSV created: {output_path}")
print(f"Total rows: {len(data)}")

print(f"CSV created: {output_path}")
print(f"Total rows: {len(data)}")
