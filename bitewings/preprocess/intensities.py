import argparse
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))

from bitewings import paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the pixel-intensity distribution of the images.')
    parser.add_argument(
        '--data-root', type=Path, default=paths.data_root(),
        help='Folder with images/ (default: $BITEWINGS_DATA_ROOT '
             f'or {paths.DEFAULT_DATA_ROOT}).',
    )
    data_root = parser.parse_args().data_root
    roots = {data_root.name: data_root / 'images'}

    bins = np.zeros((len(roots), 256))
    for i, root in enumerate(roots.values()):
        for img_path in tqdm(list(root.glob('*'))):
            img = cv2.imread(str(img_path))

            counts = np.bincount(img[..., 0].flatten(), minlength=256)
            bins[i] += counts

    clip_bins = bins

    total_count = 1_000_000
    clip_bins = clip_bins / clip_bins.sum(1, keepdims=True)
    repeats = (clip_bins * 1_000_000).astype(int)

    numbers_list = [
        np.repeat(np.arange(0, 256), reps)
        for reps in repeats
    ]

    df = pd.DataFrame({
        'Intensity value': np.concatenate(numbers_list),
        'Source': np.array(list(roots.keys())).repeat([len(n) for n in numbers_list]),
    })

    sns.kdeplot(data=df, x='Intensity value', hue='Source', bw_adjust=1)
    plt.gca().get_yaxis().set_ticks([])
    plt.savefig('kdes.png', dpi=800, bbox_inches='tight', pad_inches=None)
    plt.show()
