Use this command to change the camera view:

```gz service -s /gui/move_to/pose --reqtype gz.msgs.GUICamera --reptype gz.msgs.Boolean --timeout 2000 --req "pose: {position: {x: 0.0, y: 0.0, z: 15.0} orientation: {x: 0.0, y: 0.7071, z: 0.0, w: 0.7071}}"```

The recommended template folder from official tutorial has 4 packages. Notice that these packages have `ament_cmake`and different structures compared to Python packages defined by `ament_python`.

To place a new script under `_application`, we need to put it under `/scripts` folder, modify `package.xml`, `CMakeLists.txt`, and add `LaunchDescription` in the `.launch.py` file.

+ `ros_gz_example_application`: holds ROS 2 specific code and configurations. Namely where control, planning or any high level algoritms reside.

+ `ros_gz_example_bringup`: holds launch files and high level utilities, communication bridge between ROS and Gazebo. Any robot or hardware specific configurations go here. If we want to load a configruation file to launch the bridge node, then we should place our configuration file right inside the folder.

+ `ros_gz_example_description`: holds the SDF description of the simulated system and any other simulation assets.

+ `ros_gz_example_gazebo` :holds Gazebo specific code and configurations. Namely this is where user-defined worlds and custom system plugins end up.

Assuming that you have already installed ROS2 jazzy and Gazebo,

To run the simluation:

clone the repository.

navigate to the top level of the workspace, then

```bash
cd final_project_5527_ws
colcon build
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch ros_gz_example_bringup diff_drive.launch.py
```

Data Collection Pipeline:

```bash
gz service -s /gui/move_to/pose --reqtype gz.msgs.GUICamera --reptype gz.msgs.Boolean --timeout 2000 --req "pose: {position: {x: 0.0, y: 0.0, z: 15.0} orientation: {x: 0.0, y: 0.7071, z: 0.0, w: 0.7071}}" && python3 recorder.py
```

Diffusion Policy:

Note. It is recommended that you first run the inference node, and then run the simulation. Each time you want to restart the experiment, please re-run the inference node at first, otherwise its predictions may still be the remaining action sequences of last time. 

To run the diffsion policy model, you should create a folder `checkpoints` under `GNN_diffusion_policy_vel`, and correctly place the model weights (checkpoint) under the `checkpoints`, and then open a second terminal:

```bash
cd GNN_diffusion_policy_vel
python3 inference.py
```

Notice. We obtained our dataset in 20Hz, with 50 episodes or so. You are also welcomed to record your own dataset by going through the data collection pipeline and re-train the model. The quality of expert demonstrations does matter. 

One of the checkpoints for Graph feature concatenated with original observation (our method) is provided: https://drive.google.com/file/d/17yIItNZkzHHKuzbDiK34B5pYTXpX6dSD/view?usp=drive_link. If you have any further question, please send an Email to us and we will be happy to explain any detail of this project.
