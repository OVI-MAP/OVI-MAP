<p align="center">
  <h1 align="center">OVI-MAP: Open-Vocabulary Instance-Semantic Mapping</h1>
  <h2 align="center">CVPR 2026 Highlight✨</h2>
  <p align="center">
    <a href="https://dzl666.github.io/">Zilong Deng</a><sup>1,3</sup></span>,
    <a href="https://scholar.google.de/citations?user=TFsE4BIAAAAJ">Federico Tombari</a><sup>2,4</sup>
    <a href="https://people.inf.ethz.ch/pomarc/">Marc Pollefeys</a><sup>1,5</sup>,
    <a href="https://scholar.google.com/citations?user=dfjN3YAAAAAJ">Johanna Wald</a><sup>2,*</sup>,
    <a href="https://scholar.google.com/citations?user=U9-D8DYAAAAJ">Daniel Barath</a><sup>1,2,*</sup>
    <br>
    <sup>1</sup>ETH Zurich&emsp;
    <sup>2</sup>Google&emsp;
    <sup>3</sup>Univeristy of Zurich&emsp;
    <sup>4</sup>TU Munich&emsp;
    <sup>5</sup>Microsoft&emsp;
    <sup>*</sup>Equal contribution
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2603.26541">Paper</a> | <a href="https://ovi-map.github.io/">Project Page</a> </h3>
  <div align="center"></div>
</p>

This is the official implementation of the CVPR2026 paper **OVI-MAP**.

### BibTex

<pre><code>@misc{deng2026ovimap,
  title={OVI-MAP:Open-Vocabulary Instance-Semantic Mapping}, 
  author={Zilong Deng and Federico Tombari and Marc Pollefeys and Johanna Wald and Daniel Barath},
  year={2026},
  eprint={2603.26541},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.26541}, 
}
</code></pre>

### Timeline

- [x] 2026/05/13: Release original code & interactive visualization tools (there might be some bugs for fresh start. I am still working on it, thanks for your patience :))
- [ ] Before 2026/06/01: Bug fixing and cleaning the codebase
- [ ] Before 2026/07/01: Release the refactored codebase for easier installation and deployment.

## Installation

### Download the Repo

```bash
git clone --recurse-submodules
```

### ROS Env. (Skip this if you have it)

```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

sudo apt install curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update

sudo apt install ros-noetic-desktop-full
```

### Dependencies

```bash
sudo apt install python3-dev python3-pip python3-wstool protobuf-compiler dh-autoreconf python3-catkin-tools python3-osrf-pycommon
```

### Conda Environment

Python 3.8 (required for ROS-noetic support)

```bash
conda create -n ovi-map python==3.8.10
conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Python Packages

```bash
python -m pip install catkin_pkg rospkg rosdep rosinstall rosinstall-generator wstool
python -m pip install empy==3.3.4 pybind11 tqdm open3d scipy opencv-contrib-python==4.11.* plyfile h5py transformers==4.49.* accelerate==0.26.* sentencepiece protobuf
python -m pip install ftfy regex git+https://github.com/openai/CLIP.git
```

### Install Thrid Party Packages

```bash
export ROS_VERSION=noetic
cd semantic_mapping/mapping_ros_ws
catkin init
catkin config --extend /opt/ros/$ROS_VERSION --merge-devel
catkin config --cmake-args -DCMAKE_CXX_STANDARD=14 -DCMAKE_BUILD_TYPE=Release
wstool init src

cd src
wstool merge -t . consistent_panoptic_mapping/consistent_panoptic_mapping.rosinstall
wstool update
```

### Build the Environment

```bash
cd semantic_mapping/mapping_ros_ws/src
catkin build consistent_gsm depth_segmentation_py
```

## Dataset

Download and pre-process the dataset, put them under: **[path_to_datasets]/[dataset_name]**

### Replica

Download the Replica dataset pre-processed by [NICE-SLAM](https://pengsongyou.github.io/nice-slam).

```bash
cd [path_to_dataset]
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
unzip Replica.zip && rm Replica.zip
```

### ScanNet

You need to download the dataset youself from [ScanNet](https://kaldir.vc.in.tum.de/scannet_benchmark/documentation).
We prepared some scripts for you to extract the data you need, see: **./**


## How to Use It

### Source the environment

```bash
cd semantic_mapping
source mapping_ros_ws/devel/setup.bash
```

### Run CropFormer

Prepare your dataset and run [Cropformer](https://github.com/qqlu/Entity/blob/main/Entityv2/CODE.md).

Modify the code under **Cropformer/demo_cropformer/demo_from_dirs.py**, output the results as *.png* files:

```python
# save mask id to file
f_name= os.path.basename(path).replace('jpg', 'png')
out_filename = os.path.join(args.output, f_name)
cv2.imwrite(out_filename, mask_id)
```

For all of the experiments showed in the OVI-MAP paper, we ran Cropformer with the official *CropFormer_hornet_3x_03823a* model and the default settings:

```bash
python projects/CropFormer/demo_cropformer/demo_from_dirs.py --opts MODEL.WEIGHTS [path_to_ckpts]/CropFormer_hornet_3x_03823a.pth
```

> Model can be download under: [CropFormer_hornet_3x_03823a.pth](https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg/blob/main/CropFormer_model/Entity_Segmentation/CropFormer_hornet_3x/CropFormer_hornet_3x_03823a.pth)

Group all of RGB segmentation results by the name of scene, and place them like:

```text
[workspace dir]
  ├──geo_seg_temp/      (temp storage of the depth segmentation results, if needed)
  └──sem_seg_temp/      (temp storage of the RGB segmentation results, if needed)
        ├──office0/         (e.g.: Segmentations of the RGB images from Replica-office0)
              ├──0.jpg
              ├──xxx.jpg
              └──1999.jpg
        ├──scene0011_00/    (e.g.: Segmentations of the RGB images from ScanNet-scene0011_00)
              ├──xxx.jpg
              ├──xxx.jpg
              └──xxx.jpg
        └──[SceneName]/
```

### Run the main pipeline

```bash
bash scripts/run_replica_all.sh
```

### Run Post-Processing

#### [Step 1] Convert the instance code-book to colored instance mesh and semantic mesh

```bash
python -m scripts.utils.mesh_postprocess_utils --scene_num office0
```

#### Evaluate the instance segmentation accuracy (after Step 1)

```bash
python -m scripts.eval_inst_seg --scene_num office0
```

#### Evaluate the semantic segmentation accuracy (after Step 1)

You need to modify the scene list for your specific dataset.

```bash
python -m scripts.eval_sem_seg
```

#### Heat Map Querying (after Step 1)

```bash
python -m scripts.utils.search_heatmap --scene_num scene0011_00 --search_text sofa
```

## FAQ / Bug fixing

### Installation Problems

If you cannot build opencv when **Install Thrid Party Packages**, try to manully download these files.

```bash
cd ~/semantic_mapping/mapping_ros_ws/build/opencv3_catkin/opencv3_contrib_src/modules/xfeatures2d/src/

wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/34e4206aef44d50e6bbcd0ab06354b52e7466d26/boostdesc_bgm.i
wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/34e4206aef44d50e6bbcd0ab06354b52e7466d26/boostdesc_bgm_bi.i
wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/34e4206aef44d50e6bbcd0ab06354b52e7466d26/boostdesc_bgm_hd.i
wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/34e4206aef44d50e6bbcd0ab06354b52e7466d26/boostdesc_lbgm.i
wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/34e4206aef44d50e6bbcd0ab06354b52e7466d26/boostdesc_binboost_064.i
wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/34e4206aef44d50e6bbcd0ab06354b52e7466d26/boostdesc_binboost_128.i
wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/34e4206aef44d50e6bbcd0ab06354b52e7466d26/boostdesc_binboost_256.i

wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/fccf7cd6a4b12079f73bbfb21745f9babcd4eb1d/vgg_generated_48.i
wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/fccf7cd6a4b12079f73bbfb21745f9babcd4eb1d/vgg_generated_64.i
wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/fccf7cd6a4b12079f73bbfb21745f9babcd4eb1d/vgg_generated_80.i
wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/fccf7cd6a4b12079f73bbfb21745f9babcd4eb1d/vgg_generated_120.i
```
