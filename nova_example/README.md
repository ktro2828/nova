# NOVA Example

This example demonstrates how to use the NOVA framework to create a simple ROS2 node that publishes and subscribes to `nova_msgs::msg::CompressedVideo` messages.

## Getting Started

```shell
# Terminal 1
ros2 run nova_example nova_example_talker
```

```shell
# Terminal 2
ros2 run nova_example nova_example_listener
```

<div align="center">
    <img src="./media/nova_example.png"/>
</div>
