import torch
import torchvision.transforms as transforms
from PIL import Image
from model import ResNet50Classifier

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
MODEL_PATH = "saved_models/model_resnet50.pth"

# Image Preprocessing
IMG_SIZE = 224
prediction_transform = transforms.Compose([

    # Resize image
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    # Convert PIL image to Tensor
    transforms.ToTensor(),

    # Normalize using ImageNet statistics
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])

def load_model():

    model = ResNet50Classifier(num_classes=2)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)

    model.eval()

    return model

CLASS_NAMES = [
    "Normal",
    "OSCC"
]


def predict_image(model, image):

    # Convert uploaded image to RGB
    image = image.convert("RGB")

    # Apply preprocessing
    image_tensor = prediction_transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Move to device
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    prediction = prediction.item()

    confidence = confidence.item()

    return {

        "class_index": prediction,

        "class_name": CLASS_NAMES[prediction],

        "confidence": confidence,

        "probabilities": probabilities.cpu().numpy()[0]

    }

if __name__ == "__main__":

    model = load_model()

    print("Model Loaded Successfully!")