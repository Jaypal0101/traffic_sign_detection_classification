from flask import Flask, render_template, request
from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLO model
yolo_model = YOLO('models/best.pt')

# Load CNN model
class TrafficSignCNN(torch.nn.Module):
    def __init__(self, num_classes=43):
        super(TrafficSignCNN, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64*8*8, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

cnn_model = TrafficSignCNN(num_classes=43)
cnn_model.load_state_dict(torch.load('models/cnn_model.pth', map_location=torch.device('cpu')))
cnn_model.eval()

transform_cnn = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            image_path = filepath

            results = yolo_model.predict(filepath, conf=0.25, imgsz=416)
            img = Image.open(filepath).convert('RGB')
            w, h = img.size

            for result in results:
                for box in result.boxes:
                    x, y, bw, bh = box.xywh[0].tolist()
                    xmin = int((x - bw / 2) * w)
                    ymin = int((y - bh / 2) * h)
                    xmax = int((x + bw / 2) * w)
                    ymax = int((y + bh / 2) * h)
                    crop = img.crop((xmin, ymin, xmax, ymax))

                    cnn_input = transform_cnn(crop).unsqueeze(0)
                    with torch.no_grad():
                        output = cnn_model(cnn_input)
                        _, pred_class = output.max(1)
                        prediction = f"Detected Class: {pred_class.item()}"

    return render_template('index.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
