cmake_minimum_required(VERSION 3.8)
project(robot_interfaces)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Declare all message files
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LLMCommand.msg"
  "msg/DetectedObject.msg"
  "msg/VisionResult.msg"
  "msg/MotionStatus.msg"
  "msg/KnowledgeResult.msg"
  DEPENDENCIES std_msgs
)

ament_package()