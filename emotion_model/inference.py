import torch
import torchvision.transforms as transforms
from timm import create_model
import cv2

# select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Inference device: {device}")

# Ekman theory
emotion_labels = ['Anger', 'Happy', 'Surprise', 'Sad', 'Contempt', 'Fear', 'Disgust', 'Neutral']

# loading models
class EmotionInferencer:
    def __init__(self, model_path):
        self.model = create_model('efficientnet_b0', pretrained=False, num_classes=len(emotion_labels))
        self.model.load_state_dict(torch.load(model_path, map_location=device))
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
        """
        face_img: OpenCV format facial image（numpy array）
        return: (emotion_class, confidence)
        """
        try:
            input_tensor = self.transform(face_img)
            input_tensor = input_tensor.unsqueeze(0).to(device)  # Add batch dimension
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred_class = torch.max(probs, dim=1)
                return pred_class.item(), confidence.item()
        except Exception as e:
            print(f" Error during prediction: {e}")
            return None, None


# Create a global instance of inferencer
emotion_inferencer = EmotionInferencer("emotion_model/checkpoints/best_model_full.pt")

def predict_emotion_from_face(face_img):
    """
    External call interface, input face image, and output the predicted emotion category number and confidence level
    """
    return emotion_inferencer.predict(face_img)
