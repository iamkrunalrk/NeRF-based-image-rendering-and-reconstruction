# Efficient NeRF for Neural Scene Rendering

This project implements a **Neural Radiance Fields (NeRF)** model for rendering 3D scenes from 2D images using TensorFlow and Keras. The model focuses on volume rendering, computer vision, and neural networks to generate realistic 3D reconstructions from multi-view images. 

## Requirements

Before using this code, make sure to install the necessary dependencies:

```bash
!pip install -q --upgrade ipython==5.5.0
!pip install -q --upgrade ipykernel==4.10
```

The project also requires the following Python libraries:
- TensorFlow
- NumPy
- Matplotlib
- imageio
- tqdm

## Model

### 1. **Position Encoding (Fourier Features)**
The function `encode_position(x)` computes the Fourier feature of the input position to improve the model’s ability to handle high-frequency details in the 3D space. The mathematical formulation for encoding is:

$$
\gamma(x) = [\sin(2^0 \pi x), \cos(2^0 \pi x), \sin(2^1 \pi x), \cos(2^1 \pi x), \dots, \sin(2^{L-1} \pi x), \cos(2^{L-1} \pi x)]
$$

where:
- $x$ is the coordinate (position),
- $L$ is the number of frequencies used for encoding,
- $\sin$ and $\cos$ functions are applied at multiple scales to capture fine-grained variations.

### 2. **Ray Generation**
The function `get_rays(height, width, focal, pose)` calculates the origin and direction of rays based on camera parameters and position. It computes the rays cast from the camera and their directions in world space, based on the camera's intrinsic and extrinsic parameters.

The mathematical formulation behind ray generation is:

$$
\mathbf{o} = \mathbf{C} + \mathbf{R}(\theta) \cdot \mathbf{u}
$$

where:
- $\mathbf{o}$ is the ray's origin,
- $\mathbf{C}$ is the camera center,
- $\mathbf{R}(\theta)$ is the rotation matrix corresponding to the camera's pose $\theta$,
- $\mathbf{u}$ is the direction of the ray in camera coordinates.

### 3. **Volume Rendering**
The function `render_flat_rays(ray_origins, ray_directions, near, far, num_samples)` simulates the volumetric rendering of a scene by evaluating the color and opacity along each ray, using samples within the near and far bounds.

The volume rendering equation is given by:

$$
C(\mathbf{r}) = \int_{\mathbf{r}} T(t) \cdot \sigma(t) \cdot \mathbf{c}(t) \, dt
$$

where:
- $\mathbf{r}$ is the ray,
- $T(t)$ is the transmittance function that accounts for the accumulated opacity up to point $t$,
- $\sigma(t)$ is the volume density at point $t$,
- $\mathbf{c}(t)$ is the RGB color at point $t$.

### 4. **NeRF Model Training**
The core training process for the NeRF model is defined within the class `NeRF` by the following steps:

- **Forward Pass:** The rays are passed through the model to obtain RGB values and depth maps.
- **Loss Calculation:** The model's output is compared with the true image using the loss function. The Mean Squared Error (MSE) loss is typically used to evaluate the difference:

$$
\text{Loss} = \frac{1}{N} \sum_{i=1}^{N} ||\hat{I}_i - I_i||^2
$$

where:
- $\hat{I}_i$ is the predicted RGB value,
- $I_i$ is the true RGB value,
- $N$ is the total number of pixels in the image.

The model uses **PSNR (Peak Signal-to-Noise Ratio)** to evaluate the quality of the generated images:

$$
\text{PSNR} = 10 \cdot \log_{10} \left( \frac{\text{MAX}_I^2}{\text{MSE}} \right)
$$

where:
- $\text{MAX}_I$ is the maximum possible pixel value (e.g., 1.0 for normalized RGB).

### 5. **Training Monitor**
The `TrainMonitor` callback tracks the training progress, plotting the loss and saving the generated images at the end of each epoch. It shows the training loss, RGB output, and depth map for every training cycle.

---

## Functions

`encode_position(x)`
This function takes an input coordinate and encodes it into its corresponding Fourier feature. It helps the network understand high-frequency variations in the scene, especially for distant objects.

`get_rays(height, width, focal, pose)`
This function computes the origin point and direction of rays cast from a camera. The rays are then used to render scenes by simulating the journey of light through 3D space.

`render_flat_rays(ray_origins, ray_directions, near, far, num_samples, rand=False)`
This function performs volume rendering by computing the RGB values and depth at each ray's intersection with the scene. It simulates the way light interacts with the scene along each ray.

`get_nerf_model(num_layers, num_pos)`
Generates a neural network architecture for NeRF with the specified number of layers and positional encoding dimensions.

`render_rgb_depth(model, rays_flat, t_vals, rand=True, train=True)`
This function renders RGB images and depth maps from the NeRF model, given the flattened rays and sample points.

`TrainMonitor`
A custom callback used during training to monitor progress, save images, and plot the loss, RGB images, and depth maps after each epoch.

---

## Usage

1. **Setup Data**  
Load your dataset of multi-view images, poses, and camera intrinsics.

2. **Model Configuration**  
Create and compile the NeRF model by specifying the number of layers and positional encoding dimensions.

3. **Training**  
Train the model using the `train_step` and `test_step` functions. Monitor the progress with the `TrainMonitor` callback.

4. **Rendering**  
Use the trained model to render 3D images or depth maps by passing ray origins and directions to the `render_rgb_depth` function.


---
