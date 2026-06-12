import os
import csv
import numpy as np
from matplotlib.figure import Figure


def export_figure_to_png(figure, filepath, dpi=300):
    figure.savefig(filepath, dpi=dpi, bbox_inches='tight',
                   facecolor=figure.get_facecolor(), edgecolor='none')


def export_data_to_csv(filepath, *data_arrays, headers=None):
    data_dict = {}
    if headers is None:
        headers = [f"Column_{i+1}" for i in range(len(data_arrays))]
    for i, (header, arr) in enumerate(zip(headers, data_arrays)):
        arr = np.asarray(arr).flatten()
        if f"col_{i}" in data_dict:
            data_dict[header] = arr
        else:
            data_dict[header] = arr

    max_len = max(len(v) for v in data_dict.values()) if data_dict else 0
    for key in data_dict:
        arr = data_dict[key]
        if len(arr) < max_len:
            data_dict[key] = np.pad(arr, (0, max_len - len(arr)),
                                     constant_values=np.nan)

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(list(data_dict.keys()))
        for i in range(max_len):
            row = [data_dict[h][i] for h in data_dict.keys()]
            writer.writerow(row)


def get_export_filter():
    return "PNG Files (*.png);;CSV Files (*.csv);;All Files (*)"


def suggest_export_path(base_dir, base_name, ext):
    filename = f"{base_name}.{ext}"
    return os.path.join(base_dir, filename)
