import pandas as pd

# 1. Load the unsorted text file
# Using sep='\s+' handles any number of spaces or tabs between columns
df = pd.read_csv(
    "data/CollegeMsg.txt",
    sep="\s+",
    header=None,
    names=["source", "target", "timestamp"],
)

# 2. Sort the dataset chronologically
# This is the most critical step for temporal graphs
df = df.sort_values(by="timestamp", ascending=True).reset_index(drop=True)

# 3. Convert Unix timestamps to human-readable datetimes
# This makes it much easier for you to pick a meaningful time window
df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

# Print the overall time range so you know what dates to pick
print(f"Original dataset edges: {len(df)}")
print(f"Dataset starts at: {df['datetime'].min()}")
print(f"Dataset ends at: {df['datetime'].max()}")

# 4. Define your Temporal Window ($T_{start}$ and $T_{end}$)
# You should change these dates based on the output of the print statements above
# For example, extracting exactly 6 months of data:
T_start = "2004-04-15 00:00:00"
T_end = "2004-06-15 00:00:00"

# 5. Apply the Temporal Window filter
mask = (df["datetime"] >= T_start) & (df["datetime"] < T_end)
df_windowed = df[mask].copy()

print(f"Sampled dataset edges: {len(df_windowed)}")

# 6. Clean up and Save
# Drop the datetime column so the output matches your original 3-column format
df_windowed = df_windowed.drop(columns=["datetime"])

# Save to a new text file, separated by spaces, without column names or row numbers
df_windowed.to_csv("CollegeMsg-2m.txt", sep=" ", index=False, header=False)
print("Saved windowed dataset to 'CollegeMsg-2m.txt'")
