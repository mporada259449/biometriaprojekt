from deepface import DeepFace
import csv
import os

photos = os.listdir("test_id")

with open("resultsemotions.csv", "w") as file:
    writer = csv.writer(file)
    for photo in photos:
        result = DeepFace.analyze(img_path = f"test_id/{photo}")
        writer.writerow([photo])
        writer.writerow(result.keys())
        writer.writerow(result.values())

