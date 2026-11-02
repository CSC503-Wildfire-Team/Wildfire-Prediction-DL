# this file is the (Earth Engine + AlphaEarth embeddings)

import ee

PROJECT_ID = "wildfire-prediction-513220"   # <-- replace with your Project ID
ee.Initialize(project=PROJECT_ID)

AEF = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
BANDS = [f"A{i:02d}" for i in range(64)]

# Test: one point, year 2020
pt = ee.Geometry.Point(-121.8036, 39.0372)
img = (AEF.filterDate("2020-01-01", "2021-01-01")
          .filterBounds(pt)
          .first())

print("Band count:", len(img.bandNames().getInfo()))
print("Properties:", img.toDictionary(["UTM_ZONE", "DATASET_VERSION"]).getInfo())

# Pull the 64-number embedding at that point
vec = img.select(BANDS).reduceRegion(ee.Reducer.first(), pt, 10).getInfo()
print("First 5 values:", [round(vec[b], 3) for b in BANDS[:5]])