# Dataset: SisFall

## How to get it
1. Go to: http://sistemic.udea.edu.co/en/research/projects/english-falls/
   (SisFall is hosted by Universidad de Antioquia — search "SisFall dataset download"
   if that link has moved, it's a well-known public dataset)
2. Download the dataset zip. It's organized as folders per subject (e.g. `SA01`, `SE06`)
   — SA = young adult, SE = elderly.
3. Extract into this `data/` folder as `data/SisFall_dataset/`

## File format (know this cold before you write feature extraction)
Each `.txt` file is one trial. Columns (comma-separated, no header):
```
AccX, AccY, AccZ, GyroX, GyroY, GyroZ, AccX2, AccY2, AccZ2
```
- Sampled at 200Hz
- Accelerometer values need conversion from raw ADC to g-force (check the
  dataset's included README for the exact conversion formula — it depends on
  the sensor's resolution/range settings, usually something like:
  `acc_in_g = (2 * range_g / 2^resolution) * raw_value`)
- Filename encodes the activity: e.g. `D01_SA01_R01.txt` = Fall type D01
  (wait — actually D-prefix = ADL, F-prefix = Fall), subject SA01, repetition R01.
  **Check the dataset documentation for the exact code table** — don't guess,
  this matters for correct labeling.

## Your first task
Write a small script that:
1. Loops through all files
2. Extracts the label from the filename (F* = fall → label 1, D* = ADL → label 0)
3. Loads the signal into a pandas DataFrame
4. Prints how many fall vs non-fall trials you have (sanity check: should be
   imbalanced — far more ADL trials than fall trials in most datasets, which
   matters for how you train later — think about why, and what techniques
   handle class imbalance)
