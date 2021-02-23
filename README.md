# LumenRenderer
Standalone distribution of the Lumen path tracer

## Project description
This project has been made by a group of university students.

### Project Requirements
The requirements for the project set by the teachers are:
* Custom real-time fully ray-traced rendering aimed at next-gen AMD/NVDIA gpus such as the RTX 3080.
* HDR rendering.
* Includes animation.
* An interactive camera.
* Leveraging huge and fast SSDs.
* Supports the USD format.
* Features a small high-quality photorealistic environment.
* Uses a scene built with high-quality assets that appropriately showcase the rendering features.

These requirements are not final and can be changed if we have good reasons to do so.
The requirements can also be interpreted by us students.
For example, animation could be interpreted as simply linear motion of an object in the scene or skeletal animations.

### Goals for the project
The project allows us students to set our own goals.
Our goal for the project are:
A real-time renderer that fully renders scenes by using path-tracing and supports volumetric rendering.

### Duration of the project.
This project is a year long project for third-year students.
The project will be worked on throughout the year when there are enough students to work on this project.
The year is divided up into 4 blocks of 8 working weeks.
Each block there is chance for the students to switch between on-going projects.
However, it is recommended and somewhat expected by the teachers to stay with your project for the entire year.

### Project features and systems
#### • Wavefront algorithm
  The wavefront algorithm is an algorithm that aims to utilize the parallel computation potential to calculate shading.
  To do this we combine the Optix SDK and the CUDA language.
  We use the Optix SDK to utilize the ray tracing capabilities of the GPU.
    Optix will trace the rays that we need in order to calculate visibility and shading of intersection points.
  We use the CUDA language to utilize the parallel computation capabilities.
    With CUDA we write kernels that run on the GPU and generate ray definitions and calculate shading values.

## Contributions
### Main contributors

| Name | Concepting | Pre-production | Production | Release |
| :--- | :---: | :---: | :---: | :---: |
| **Person 1** | :heavy_check_mark: | :x: | :x: |  |
| **Person 2** | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  |
| **Person 3** | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  |
| **Person 4** | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  |
| **Person 5** | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  |
| **Person 6** | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  |
| **Person 7** | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  |

#### Contributions per contributor
* **Person 1**
  * [**Wavefront algorithm**](./README.md#-wavefront-algorithm)

### Additional contributors

* **Person 1:**
  - Contribution 1
 

## Getting started with the project.
### Prerequisites:
* [CMake (minimal required version: 3.16)](https://cmake.org/download/) 

### Recommended tools:
* [Visual Studio 2019](https://visualstudio.microsoft.com/)

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
| **USE_WAVEFRONT** | Whether to use the wavefront rendering pipeline or not. The wavefront rendering pipeline is a is the main rendering pipeline and probably has better support than the old rendering pipeline. This option adds the WAVEFRONT preprocessor defintions to the `Sandbox` and `LumenPT` projects. |

#### Notes
* The root folder is by default: `<Path to download folder>/Lumen_Engine`. It is the first directory containing a `CMakeLists.txt` file.
* If you place the project solution in the same folder as the root folder it supposed to work, however this has not been tested extensively.
  This means that selecting another folder is recommended.
  CMake documentation refers to these methods as `in-source` and `out-of-source`.
* The documentation for using the CMake command line can be found here: https://cmake.org/cmake/help/latest/manual/cmake.1.html
* Tutorials for using CMake can be found here: https://cmake.org/cmake/help/latest/guide/tutorial/index.html
