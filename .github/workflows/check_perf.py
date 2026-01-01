import csv
import sys

THRESHOLD = {
    "sobel": 30.0,      # GPixel/s
    "gaussian": 40.0,
}

with open("bench.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row["filter"]
        if row["gpixel_s"] == "SKIP":
            print(f"{name}: skipped")
            continue

        perf = float(row["gpixel_s"])
        if perf < THRESHOLD[name]:
            print(f"{name} regression: {perf:.2f} < {THRESHOLD[name]}")
            sys.exit(1)

        print(f"{name}: {perf:.2f} GPixel/s")

print("Performance OK")
