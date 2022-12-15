import os
import csv
from deepface import DeepFace
photos = os.listdir("Arnold_Schwarzenegger")
original = "Arnold_Schwarzenegger_0003.jpg"
with open("resultsarnold.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerow(["verified", "distance", "threshold", "model", "detector_backend", "similarity_metric"])
    for photo in photos:
        writer.writerow([photo])
        try:
            res = DeepFace.verify(img1_path=original, img2_path=f"Arnold_Schwarzenegger/{photo}", model_name="DeepFace")
            writer.writerow(res.values())
        except ValueError:
            writer.writerow(["That is not a face"])

           