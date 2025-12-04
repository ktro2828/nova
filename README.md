# NOVA

This project is a refactoring of the [tier4/accelerated_image_processor](https://github.com/tier4/accelerated_image_processor).

## Getting Started

```bash
git clone https://github.com/ktro2828/nova.git
cd nova
rosdep update
rosdep install -y --from-path . --ignore-src --rosdistro $ROS_DISTRO
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```
