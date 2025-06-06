import os
import torch
import torchvision.transforms as transforms
from timm import create_model
import cv2

# select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Inference device: {device}")

# Ekman theory
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
# loading models
class EmotionInferencer:
    def __init__(self, model_name="efficientnet_b0", checkpoint_name="best_model_efficientnet_b0.pt"):
        """
        model_name: (efficientnet_b0, mobilenetv2_100, resnet50)
        checkpoint_name:
        """
        # structural modeling
        self.model = create_model(model_name, pretrained=False, num_classes=len(emotion_labels))
        # Loading weight
        checkpoint_path = os.path.join("checkpoints", checkpoint_name)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # OpenCV image -> PIL image
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, face_img):
        try:
            input_tensor = self.transform(face_img)
            input_tensor = input_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred_class = torch.max(probs, dim=1)
                return pred_class.item(), confidence.item()
        except Exception as e:
            print(f" Error during prediction: {e}")
            return None, None

# Usage Examples
# Run in the emotion_model directory and assume that the image is in OpenCV format
# Example:
# face_img = cv2.imread('test_face.jpg')[:, :, ::-1] # BGR to RGB
# inferencer = EmotionInferencer(model_name="resnet50", checkpoint_name="best_model_resnet50.pt")
# pred_id, conf = inferencer.predict(face_img)
# print(emotion_labels[pred_id], conf)

# Recommended external interface
def predict_emotion_from_face(face_img, model_name="efficientnet_b0", checkpoint_name="best_model_efficientnet_b0.pt"):
    """
    Input the OpenCV face image and output the predicted expression category number and confidence level
    The model type and weight
    """
    inferencer = EmotionInferencer(model_name=model_name, checkpoint_name=checkpoint_name)
    return inferencer.predict(face_img)
