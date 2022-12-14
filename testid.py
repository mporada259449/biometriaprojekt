from deepface import DeepFace
import os
import csv

photos = os.listdir("test_id")
with open("resultsid.csv", "w") as file:
    writer = csv.writer(file)
    for photo in photos:
        result = DeepFace.find(img_path=f"test_id/{photo}", db_path="face_dataset", model_name="DeepFace")
        writer.writerow([photo])
        writer.writerow(result["identity"].values.tolist())


