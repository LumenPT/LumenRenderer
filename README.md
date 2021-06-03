# LumenRenderer

___
**(The Lumen Renderer is not related to or affiliated with Unreal Engine 5's Lumen system.)**
___

Standalone distribution of the Lumen path tracer.
This is a student project that is being made over the course of 1 year by 6 students of Breda University of Applied Sciences (BUAS).
Our goal is a real-time (30fps+) path tracer with support for volumetric bodies.

![MicrosoftTeams-image (12)](https://user-images.githubusercontent.com/9714482/120190774-591d5e80-c219-11eb-8d01-3d9b8d7599f8.png)

## Project description
The project is made from the ground up using C++, CUDA 10 and OptiX 7.1. A DirectX11 layer is also used for interaction with external libraries that do not natively support CUDA. Libraries such as DLSS (Nvidia's Deep Learning Super Sampling) and NRD (Nvidia's Real-Time Denoiser). We are targetting 30fps+ at 1440p resolutions on RTX3090 GPU's.

### Project Requirements/Goals
* Custom real-time fully path-traced rendering on RTX 3090.
* HDR rendering.
* Support for volumetrics (OpenVDB/NanoVDB files).
* An interactive camera.
* Features high-quality denoising (NRD).
* Features a small high-quality photorealistic environment.
* The Wavefront algorithm is implemented for our path-tracer.
* The ReSTIR algorithm is implemented for our path-tracer.

### Goals for the project
The project allows us students to set our own goals.
Our goal for the project are:
A real-time renderer that fully renders scenes by using path-tracing and supports volumetric rendering.

### Project features and systems
#### • Wavefront algorithm
  The wavefront algorithm is an algorithm that aims to utilize the parallel computation potential to calculate shading.
  To do this we combine the Optix SDK and the CUDA language.
  We use the Optix SDK to utilize the ray tracing capabilities of the GPU.
    Optix will trace the rays that we need in order to calculate visibility and shading of intersection points.
  We use the CUDA language to utilize the parallel computation capabilities.
    With CUDA we write kernels that run on the GPU and generate ray definitions and calculate shading values.

## Getting started with the project.
### Prerequisites:
* [CMake (minimal required version: 3.20)](https://cmake.org/download/) 
* [CUDA toolkit](https://developer.nvidia.com/cuda-downloads)
* [OptiX 7 SDK](https://developer.nvidia.com/designworks/optix/download)
### Optional
* [Access to DLSS](https://developer.nvidia.com/dlss)
* [Access to NRD](https://developer.nvidia.com/nvidia-rt-denoiser)

### Recommended tools:
* [Visual Studio 2019 (version 16.6.3)](https://visualstudio.microsoft.com/)
Note that Visual Studio versions above 16.6.3 are not guarranteed to work due to an unknown bug. If you followed all the build steps correctly and the project still does not compile check your VS version as you may need to downgrade. We are looking into the issue to fix it.

### Build Instructions using CMake
* **Download the project.**
* **To use the CMake-GUI**
  1. Open the CMake-GUI application.
  2. Select the path to the root folder of the project.
  3. Select the path to the folder where you want to place the project solution.
  4. Press configure and select the options that you want to use.
  5. Press configure again and then press generate.
  6. Now there should be a project solution in the selected folder.
* **To use the command-line**
  1. Open the command line.
  2. Type the command: `cmake [options] -S <path to root folder> -B <path to project solution folder>`.
     Replace the `<path to root folder>` with the path to the root folder of the project.
     Replace the `<path to project solution folder>` with the path to the folder where you want to place the project solution.
     Optionally the `[options]` with the options that you want to use to generate the project solution.
     These need to be in the format: `-D<option>:<type>=<value>` or `-D<option>=<value>`,
     where you replace `<option>` with the option name and `<value>` with the value you want the option to be. 
     Optionally you can also specify a type by replacing `<type>`.
  3. After executing the command the project solution should be in the selected folder.

#### Options
| Option | Description |
| :--- | :--- |
| **WIP_USE_DLSS** | Whether to use DLSS or not. DLSS is closed-source, so building with DLSS will require you to have access to repository and to manually move the appropriate DLSS files in the correct vendor folder. |
| **WIP_USE_NRD** | Whether to use NRD or not. NRD is closed-source, so building with NRD will require you to have access to repository and to manually move the appropriate DLSS files in the correct vendor folder. |

#### Notes
* The root folder is by default: `<Path to download folder>/Lumen_Engine`. It is the first directory containing a `CMakeLists.txt` file.
* If you place the project solution in the same folder as the root folder it supposed to work, however this has not been tested extensively.
  This means that selecting another folder is recommended.
  CMake documentation refers to these methods as `in-source` and `out-of-source`.
* The documentation for using the CMake command line can be found here: https://cmake.org/cmake/help/latest/manual/cmake.1.html
* Tutorials for using CMake can be found here: https://cmake.org/cmake/help/latest/guide/tutorial/index.html

## Gallery
### Colored specular highlights
![unknown](https://user-images.githubusercontent.com/9714482/120173799-b0fd9a80-c204-11eb-9c10-1e12e065890f.png)

### Prototype for homogenous volumetrics
![unknown (1)](https://user-images.githubusercontent.com/9714482/120174052-f4580900-c204-11eb-984b-1b6ba9a9e5e9.png)

### Realistic soft shadows
![MicrosoftTeams-image (12)](https://user-images.githubusercontent.com/9714482/120172177-105aab00-c203-11eb-81ef-0d3046d9bd9f.png)
