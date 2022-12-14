from deepface import DeepFace
import csv
result = DeepFace.find(img_path="StefanKarl.jpg", db_path="face_dataset", model_name="DeepFace", enforce_detection=False)
result  = result["identity"].values.tolist()
with open("resultsintrusion.csv", "w") as file:
    writer = csv.writer(file)
    for id in result:
        try:
            res = DeepFace.verify(img1_path="StefanKarl.jpg", img2_path=id, model_name="DeepFace")
            writer.writerow([id])
            writer.writerow(["verified", "distance", "threshold", "model", "detector_backend", "similarity_metric"])
            writer.writerow(res.values())
        except ValueError:
            writer.writerow(["That is not a face"])
        
    