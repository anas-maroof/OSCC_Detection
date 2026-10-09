import cv2
import torch
import numpy as np

from PIL import Image

from torchvision import transforms

from pytorch_grad_cam import GradCAM

from pytorch_grad_cam.utils.image import show_cam_on_image

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMG_SIZE = 224


prediction_transform = transforms.Compose([

    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])


def generate_gradcam(model, image):

    """
    Returns:
        original_image
        heatmap
        overlay_image
    """

    image = image.convert("RGB")

    original_image = np.array(
        image.resize((224,224))
    ) / 255.0

    input_tensor = prediction_transform(image)

    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    model.eval()

    with torch.no_grad():

        outputs = model(input_tensor)

        prediction = outputs.argmax(dim=1).item()

    # Last convolution layer of ResNet50

    target_layers = [
        model.backbone.layer4[-1]
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(

        input_tensor=input_tensor,

        targets=[
            ClassifierOutputTarget(prediction)
        ]

    )[0]

    overlay = show_cam_on_image(

        original_image.astype(np.float32),

        grayscale_cam,

        use_rgb=True

    )

    heatmap = cv2.applyColorMap(

        np.uint8(grayscale_cam*255),

        cv2.COLORMAP_JET

    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    return (

        (original_image*255).astype(np.uint8),

        heatmap,

        overlay

    )