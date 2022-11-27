// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/modules/face_geometry/protos/face_geometry.pb.h"
#include "mediapipe/modules/face_geometry/libs/mesh_3d_utils.h"

#include <zmq.h>
#include <zmq_utils.h>

struct Vector3d
{
  Vector3d()
  {
    x = 0;
    y = 0;
    z = 0;
  }
  Vector3d(float _x, float _y, float _z)
  {
    x = _x;
    y = _y;
    z = _z;
  }
  float x;
  float y;
  float z;
};

struct FaceMesh
{
  int32_t id;
  float capture_width;
  float capture_height;
  Vector3d landmarkers[478];
};

#define GL_TRIANGLES                      0x0004

struct RenderableMesh3d {
  static absl::StatusOr<RenderableMesh3d> CreateFromProtoMesh3d(
      const mediapipe::face_geometry::Mesh3d& proto_mesh_3d) {
    mediapipe::face_geometry::Mesh3d::VertexType vertex_type = proto_mesh_3d.vertex_type();

    RenderableMesh3d renderable_mesh_3d;
    renderable_mesh_3d.vertex_size = GetVertexSize(vertex_type);
    ASSIGN_OR_RETURN(
        renderable_mesh_3d.vertex_position_size,
        GetVertexComponentSize(vertex_type, mediapipe::face_geometry::VertexComponent::POSITION),
        _ << "Failed to get the position vertex size!");
    ASSIGN_OR_RETURN(
        renderable_mesh_3d.tex_coord_position_size,
        GetVertexComponentSize(vertex_type, mediapipe::face_geometry::VertexComponent::TEX_COORD),
        _ << "Failed to get the tex coord vertex size!");
    ASSIGN_OR_RETURN(
        renderable_mesh_3d.vertex_position_offset,
        GetVertexComponentOffset(vertex_type, mediapipe::face_geometry::VertexComponent::POSITION),
        _ << "Failed to get the position vertex offset!");
    ASSIGN_OR_RETURN(
        renderable_mesh_3d.tex_coord_position_offset,
        GetVertexComponentOffset(vertex_type, mediapipe::face_geometry::VertexComponent::TEX_COORD),
        _ << "Failed to get the tex coord vertex offset!");

    switch (proto_mesh_3d.primitive_type()) {
      case mediapipe::face_geometry::Mesh3d::TRIANGLE:
        renderable_mesh_3d.primitive_type = GL_TRIANGLES;
        break;

      default:
        RET_CHECK_FAIL() << "Only triangle primitive types are supported!";
    }

    renderable_mesh_3d.vertex_buffer.reserve(
        proto_mesh_3d.vertex_buffer_size());
    for (float vertex_element : proto_mesh_3d.vertex_buffer()) {
      renderable_mesh_3d.vertex_buffer.push_back(vertex_element);
    }

    renderable_mesh_3d.index_buffer.reserve(proto_mesh_3d.index_buffer_size());
    for (uint32_t index_element : proto_mesh_3d.index_buffer()) {
      RET_CHECK_LE(index_element, std::numeric_limits<uint16_t>::max())
          << "Index buffer elements must fit into the `uint16` type in order "
             "to be renderable!";

      renderable_mesh_3d.index_buffer.push_back(
          static_cast<uint16_t>(index_element));
    }

    return renderable_mesh_3d;
  }

  uint32_t vertex_size;
  uint32_t vertex_position_size;
  uint32_t tex_coord_position_size;
  uint32_t vertex_position_offset;
  uint32_t tex_coord_position_offset;
  uint32_t primitive_type;

  std::vector<float> vertex_buffer;
  std::vector<uint16_t> index_buffer;
};


constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";
constexpr char kOutputLandMarks[] = "landmarks";
constexpr char kMultiFaceLandMarks[] = "multi_face_landmarks";
constexpr char kFaceRectsLandMarks[] = "face_rects_from_landmarks";
constexpr char kMultiFaceGeometry[] = "multi_face_geometry";


ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "",
          "Full path of video to load. "
          "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "",
          "Full path of where to save result (.mp4 only). "
          "If not provided, show result in a window.");


  static absl::StatusOr<std::array<float, 16>>
  Convert4x4MatrixDataToArrayFormat(const mediapipe::MatrixData& matrix_data) {
    RET_CHECK(matrix_data.rows() == 4 &&  //
              matrix_data.cols() == 4 &&  //
              matrix_data.packed_data_size() == 16)
        << "The matrix data must define a 4x4 matrix!";

    std::array<float, 16> matrix_array;
    for (int i = 0; i < 16; i++) {
      matrix_array[i] = matrix_data.packed_data(i);
    }

    // Matrix array must be in the OpenGL-friendly column-major order. If
    // `matrix_data` is in the row-major order, then transpose.
    if (matrix_data.layout() == mediapipe::MatrixData::ROW_MAJOR) {
      std::swap(matrix_array[1], matrix_array[4]);
      std::swap(matrix_array[2], matrix_array[8]);
      std::swap(matrix_array[3], matrix_array[12]);
      std::swap(matrix_array[6], matrix_array[9]);
      std::swap(matrix_array[7], matrix_array[13]);
      std::swap(matrix_array[11], matrix_array[14]);
    }

    return matrix_array;
  }

absl::Status RunMPPGraph() {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      absl::GetFlag(FLAGS_calculator_graph_config_file),
      &calculator_graph_config_contents));
  LOG(INFO) << "Get calculator graph config contents: "
            << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  /*TBD ELI Zeromq*/
  void *context = zmq_ctx_new();
  void *zmqPublisher = zmq_socket(context, ZMQ_PUB);
  int rc = zmq_bind(zmqPublisher, "tcp://*:5556");
  assert(rc == 0);

  LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Initialize the camera or load the video.";
  cv::VideoCapture capture;
  const bool load_video = !absl::GetFlag(FLAGS_input_video_path).empty();
  if (load_video) {
    capture.open(absl::GetFlag(FLAGS_input_video_path));
  } else {
    capture.open(0);
  }
  RET_CHECK(capture.isOpened());

  cv::VideoWriter writer;
  const bool save_video = !absl::GetFlag(FLAGS_output_video_path).empty();
  if (!save_video) {
    cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 30);
#endif
  }

  LOG(INFO) << "Start running the calculator graph.";
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                   graph.AddOutputStreamPoller(kOutputStream));

  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller landMarksPoller,
                   graph.AddOutputStreamPoller(kMultiFaceLandMarks));


  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller multiFaceGeometryPoller,
                   graph.AddOutputStreamPoller(kMultiFaceGeometry));
  //
    
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  float capture_width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
  float capture_height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

  LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;
  while (grab_frames) {
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) {
      if (!load_video) {
        LOG(INFO) << "Ignore empty frames from camera.";
        continue;
      }
      LOG(INFO) << "Empty frame, end of video reached.";
      break;
    }
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    if (!load_video) {
      cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
    }

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Send image packet into the graph.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        kInputStream, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(frame_timestamp_us))));

    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;
    if (!poller.Next(&packet)) break;

    //
    // Get the packet containing multi_hand_landmarks.
    if (landMarksPoller.QueueSize() > 0)
    {
      ::mediapipe::Packet mesh_landmarks_packet;
      if (!landMarksPoller.Next(&mesh_landmarks_packet))
        break;
      const auto &landmarks =
          mesh_landmarks_packet.Get<
              std::vector<::mediapipe::NormalizedLandmarkList>>();

      int face_mesh_index = 0;
      FaceMesh faceMesh = {0};
      faceMesh.capture_width = capture_width ;
      faceMesh.capture_height = capture_height;
      for (const auto &single_mesh_face_landmarks : landmarks)
      {
        faceMesh.id = face_mesh_index++;
        for (int i = 0; i < single_mesh_face_landmarks.landmark_size(); ++i)
        {
          if(i>=478)break;// not needed , just for safety
          const auto &landmark = single_mesh_face_landmarks.landmark(i);
          faceMesh.landmarkers[i] = Vector3d(landmark.x(),landmark.y(),landmark.z());
        }

        zmq_send (zmqPublisher, (const unsigned char*)&faceMesh, sizeof(FaceMesh), 0);
      }

    }

    if (multiFaceGeometryPoller.QueueSize() > 0)
    {
      ::mediapipe::Packet face_geometry_packet;
      if (!multiFaceGeometryPoller.Next(&face_geometry_packet))
        break;

      const auto &multi_face_geometry = face_geometry_packet.Get<std::vector<mediapipe::face_geometry::FaceGeometry>>();
      const int num_faces = multi_face_geometry.size();

      std::vector<std::array<float, 16>> face_pose_transform_matrices(num_faces);
      std::vector<RenderableMesh3d> renderable_face_meshes(num_faces);

      for (int i = 0; i < num_faces; ++i)
      {
        const auto &face_geometry = multi_face_geometry[i];

        ASSIGN_OR_RETURN(
            face_pose_transform_matrices[i],
            Convert4x4MatrixDataToArrayFormat(
                face_geometry.pose_transform_matrix()),
            _ << "Failed to extract the face pose transformation matrix!");

        // column major matrix
        // LOG(INFO) << "face_pose_transform_matrix "<< i;
        // LOG(INFO) << face_pose_transform_matrices[i][0] << ":" << face_pose_transform_matrices[i][4] << ":" << face_pose_transform_matrices[i][8]<< ":" << face_pose_transform_matrices[i][12];
        // LOG(INFO) << face_pose_transform_matrices[i][1] << ":" << face_pose_transform_matrices[i][5] << ":" << face_pose_transform_matrices[i][9]<< ":" << face_pose_transform_matrices[i][13];
        // LOG(INFO) << face_pose_transform_matrices[i][2] << ":" << face_pose_transform_matrices[i][6] << ":" << face_pose_transform_matrices[i][10]<< ":" << face_pose_transform_matrices[i][14];
        // LOG(INFO) << face_pose_transform_matrices[i][3] << ":" << face_pose_transform_matrices[i][7] << ":" << face_pose_transform_matrices[i][11]<< ":" << face_pose_transform_matrices[i][15];

        // Extract the face mesh as a renderable.
         ASSIGN_OR_RETURN(
          renderable_face_meshes[i],
          RenderableMesh3d::CreateFromProtoMesh3d(face_geometry.mesh()),
          _ << "Failed to extract a renderable face mesh!");

          // LOG(INFO) << "RenderableMesh3d " << i 
          // << " vertex_size:" << renderable_face_meshes[i].vertex_size 
          // << " vertex_position_size:" << renderable_face_meshes[i].vertex_position_size
          // << " tex_coord_position_size:" << renderable_face_meshes[i].tex_coord_position_size 
          // << " vertex_position_offset:" << renderable_face_meshes[i].vertex_position_offset 
          // << " tex_coord_position_offset:" << renderable_face_meshes[i].tex_coord_position_offset 
          // << " vertex_buffer size:" << renderable_face_meshes[i].vertex_buffer.size() 
          // << " index_buffer size:" << renderable_face_meshes[i].index_buffer.size() 
          // ;
      }
    }

    auto& output_frame = packet.Get<mediapipe::ImageFrame>();

    // Convert back to opencv for display or saving.
    cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
    if (save_video) {
      if (!writer.isOpened()) {
        LOG(INFO) << "Prepare video writer.";
        writer.open(absl::GetFlag(FLAGS_output_video_path),
                    mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
                    capture.get(cv::CAP_PROP_FPS), output_frame_mat.size());
        RET_CHECK(writer.isOpened());
      }
      writer.write(output_frame_mat);
    } else {
      cv::imshow(kWindowName, output_frame_mat);
      // Press any key to exit.
      const int pressed_key = cv::waitKey(5);
      if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
    }
  }

  LOG(INFO) << "Shutting down.";

  zmq_close (zmqPublisher);
  zmq_ctx_destroy (context);

  if (writer.isOpened()) writer.release();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  absl::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
