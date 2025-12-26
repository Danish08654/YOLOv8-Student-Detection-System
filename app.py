from ultralytics import YOLO
import cv2
import os

# === PART 1: TRAIN THE MODEL ===

print("📦 Training YOLOv8 model...")

model = YOLO('yolov8n.pt')  # Load nano model (smallest & fastest)

model.train(
    data='data.yaml',         # ✅ Path to your YAML file (in same dir as app.py)
    epochs=50,
    imgsz=640,
    batch=16,
    name='student_detector',
    workers=4
)

print("\n✅ Training completed.")
print("🎯 Best weights saved to: runs/detect/student_detector/weights/best.pt")

# === PART 2: LOAD IMAGE AND COUNT STUDENTS ===

print("\n🔍 Loading trained model...")
trained_model = YOLO('runs/detect/student_detector/weights/best.pt')

# Path to the test image
image_path = 'test_image.jpg'  # Put a test image in the same folder

if not os.path.exists(image_path):
    print(f"\n❌ Image '{image_path}' not found. Please add a test image.")
else:
    print(f"\n🖼️ Running prediction on: {image_path}")
    results = trained_model.predict(source=image_path, conf=0.4, save=True)

    # Count detected students
    student_count = len(results[0].boxes)

    print(f"\n🎓 Number of students detected: {student_count}")

    # Display result image
    result_img_path = 'runs/detect/predict/image0.jpg'
    if os.path.exists(result_img_path):
        img = cv2.imread(result_img_path)
        cv2.imshow("Predicted Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("⚠️ Prediction image not found.")
