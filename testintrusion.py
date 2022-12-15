from deepface import DeepFace
import csv
import os
photos = os.listdir("intruders")

with open("resultsintrusion.csv", "w") as file:
    writer = csv.writer(file)
    for photo in photos:
        writer.writerow([photo])
        result = DeepFace.find(img_path=f"intruders/{photo}", db_path="face_dataset", model_name="DeepFace")
        result = result["identity"].values.tolist()
        for id in result:
            try:
                res = DeepFace.verify(img1_path=f"intruders/{photo}", img2_path=id, model_name="DeepFace")
                writer.writerow([id])
                writer.writerow(["verified", "distance", "threshold", "model", "detector_backend", "similarity_metric"])
                writer.writerow(res.values())
            except ValueError:
                writer.writerow(["That is not a face"])
        
    