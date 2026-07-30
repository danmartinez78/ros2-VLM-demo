// Copyright 2025 cosmos_ros2_video_reasoner contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "cosmos_ros2_video_reasoner/cosmos_reasoner_node.hpp"
#include "cosmos_ros2_video_reasoner/tensorrt_edge_llm_backend.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto options = rclcpp::NodeOptions{};

  // Read engine paths from ROS parameters before constructing the node so
  // that we can build the backend config.  The node will redeclare them
  // during construction; we pre-read here only to pass to the backend.
  rclcpp::Node::SharedPtr param_node =
    std::make_shared<rclcpp::Node>("cosmos_reasoner", options);

  param_node->declare_parameter("llm_engine_dir", "");
  param_node->declare_parameter("multimodal_engine_dir", "");
  param_node->declare_parameter("edge_llm_plugin_path", "");
  param_node->declare_parameter("jpeg_quality", 90);

  cosmos_ros2_video_reasoner::TensorRTEdgeLLMConfig backend_config;
  backend_config.llm_engine_dir =
    param_node->get_parameter("llm_engine_dir").as_string();
  backend_config.multimodal_engine_dir =
    param_node->get_parameter("multimodal_engine_dir").as_string();
  backend_config.edge_llm_plugin_path =
    param_node->get_parameter("edge_llm_plugin_path").as_string();
  backend_config.jpeg_quality =
    static_cast<int>(param_node->get_parameter("jpeg_quality").as_int());

  param_node.reset();  // no longer needed

  auto backend = std::make_unique<
    cosmos_ros2_video_reasoner::TensorRTEdgeLLMBackend>(backend_config);

  auto node = std::make_shared<cosmos_ros2_video_reasoner::CosmosReasonerNode>(
    std::move(backend), options);

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
