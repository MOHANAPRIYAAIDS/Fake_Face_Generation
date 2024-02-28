# Import necessary libraries
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

app = Flask(__name__)

# Define the deconv function
def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization."""
    # sequence of transpose + optional batch norm layers
    layers = []
    transpose_conv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    # append transpose convolutional layer
    layers.append(transpose_conv_layer)

    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self, z_size, conv_dim):
        super(Generator, self).__init__()

        self.conv_dim = conv_dim

        # first, fully-connected layer
        self.fc = nn.Linear(z_size, conv_dim * 8 * 2 * 2)

        # transpose conv layers
        self.t_conv1 = deconv(conv_dim * 8, conv_dim * 4)
        self.t_conv2 = deconv(conv_dim * 4, conv_dim * 2)
        self.t_conv3 = deconv(conv_dim * 2, conv_dim)
        self.t_conv4 = deconv(conv_dim, 3, batch_norm=False)

    def forward(self, x):
        out = self.fc(x)
        out = out.view(-1, self.conv_dim * 8, 2, 2)
        out = F.relu(self.t_conv1(out))
        out = F.relu(self.t_conv2(out))
        out = F.relu(self.t_conv3(out))
        out = self.t_conv4(out)
        out = F.tanh(out)
        return out

# Load the trained GAN model
g_conv_dim = 32
z_size = 100
G = Generator(z_size=z_size, conv_dim=g_conv_dim)
G_state_dict = torch.load("<your path to file>")
G.load_state_dict(G_state_dict)
G.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Generate a new random seed for the noise vector using the current timestamp
    seed = int(time.time())
    np.random.seed(seed)
    z = np.random.uniform(-1, 1, size=(1, z_size))
    z = torch.from_numpy(z).float()

   # Generate image using GAN
    fake_image = G(z)

    # Post-process the generated image
    fake_image = fake_image.detach().cpu().numpy().squeeze()
    fake_image = np.transpose(fake_image, (1, 2, 0))
    fake_image = ((fake_image + 1) * 255 / 2).astype(np.uint8)

# Save the generated image with a dynamic filename based on the seed
    seed = int(time.time())
    image_filename = f'generated_image_{seed}.png'
    image_path = f'your path to file'
    pil_image = Image.fromarray(fake_image)
    pil_image.save(image_path)

    return render_template('result.html', image_path=f'/templates/{image_filename}', image_filename=image_filename)



@app.route('/templates/<filename>')
def send_image(filename):
    return send_from_directory("templates", filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)  
