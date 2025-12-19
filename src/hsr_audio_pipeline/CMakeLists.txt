cmake_minimum_required(VERSION 3.0.2)
project(hsr_audio_pipeline)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  audio_common_msgs
  smach
  smach_ros
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs audio_common_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Pythonノードのインストール設定
catkin_install_python(
  PROGRAMS
    scripts/audio_rms_monitor.py
    scripts/silero_vad_node.py
    scripts/faster_whisper_asr_node.py
    scripts/faster_whisper_asr_simple_node.py
    scripts/gpsr_intent_node.py
    scripts/asr_plain_echo.py
    scripts/gpsr_smach_node.py
    scripts/gpsr_parser_node.py
    scripts/asr_plain_echo.py
    scripts/gpsr_intent_echo.py
    scripts/apply_corrections_example.py
    scripts/corrections_candidates.py
    scripts/gen_corrections.py    
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

