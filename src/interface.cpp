#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   // std::vector, std::pair, std::array, std::unordered_map, std::tuple
#include <pybind11/numpy.h> // py::array_t
#include <opencv2/opencv.hpp>

#include "infer/sam3infer.hpp"
#include "infer/sam3_tracker.hpp"
#include "common/object.hpp"
#include "osd/osd.hpp" // 【新増】包含 OSD 头文件

namespace py = pybind11;

// --- 辅助函数：cv::Mat <-> numpy ---
// 将 cv::Mat 转换为 numpy array (拷贝数据)
py::array_t<uint8_t> mat_to_numpy(const cv::Mat &mat)
{
    if (mat.empty())
        return py::array_t<uint8_t>();

    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides;

    if (mat.channels() == 1)
    {
        shape = {mat.rows, mat.cols};
        strides = {mat.step[0], mat.step[1]};
    }
    else
    {
        shape = {mat.rows, mat.cols, mat.channels()};
        strides = {mat.step[0], mat.step[1], mat.elemSize1()};
    }
    return py::array_t<uint8_t>(shape, strides, mat.data);
}

// 将 numpy array 转换为 cv::Mat (共享内存，修改 Mat 会影响 numpy)
cv::Mat numpy_to_mat(py::array_t<uint8_t> &input)
{
    py::buffer_info buf = input.request();
    if (buf.ndim != 2 && buf.ndim != 3)
    {
        throw std::runtime_error("Input numpy array must be 2D or 3D.");
    }
    int rows = buf.shape[0];
    int cols = buf.shape[1];
    int channels = (buf.ndim == 3) ? buf.shape[2] : 1;
    int type = (channels == 1) ? CV_8UC1 : CV_8UC3;
    
    // 注意：这里使用 buf.ptr 创建 Mat，意味着与 numpy 共享内存
    return cv::Mat(rows, cols, type, buf.ptr, buf.strides[0]);
}

PYBIND11_MODULE(trtsam3, m)
{
    m.doc() = "Python bindings for Sam3Infer (One Vision, Many Prompts) using pybind11";

    // --- Enum 绑定 ---
    py::enum_<object::ObjectType>(m, "ObjectType")
        .value("UNKNOW", object::ObjectType::UNKNOW)
        .value("DETECTION", object::ObjectType::DETECTION)
        .value("POSE", object::ObjectType::POSE)
        .value("OBB", object::ObjectType::OBB)
        .value("SEGMENTATION", object::ObjectType::SEGMENTATION)
        .export_values();

    // --- 基础结构绑定 ---
    py::class_<object::Box>(m, "Box")
        .def(py::init<float, float, float, float>(),
             py::arg("left") = 0, py::arg("top") = 0, py::arg("right") = 0, py::arg("bottom") = 0)
        .def_readwrite("left", &object::Box::left)
        .def_readwrite("top", &object::Box::top)
        .def_readwrite("right", &object::Box::right)
        .def_readwrite("bottom", &object::Box::bottom)
        .def("__repr__", [](const object::Box &b)
             { return "<Box l=" + std::to_string(b.left) + ", t=" + std::to_string(b.top) +
                      ", r=" + std::to_string(b.right) + ", b=" + std::to_string(b.bottom) + ">"; });

    py::class_<object::Segmentation>(m, "Segmentation")
        .def(py::init<>())
        .def_property("mask", [](object::Segmentation &self)
                      { return mat_to_numpy(self.mask); }, [](object::Segmentation &self, py::array_t<uint8_t> array)
                      { self.mask = numpy_to_mat(array).clone(); });

    py::class_<object::DetectionBox>(m, "DetectionBox")
        .def(py::init<>())
        .def_readwrite("box", &object::DetectionBox::box)
        .def_readwrite("score", &object::DetectionBox::score)
        .def_readwrite("class_id", &object::DetectionBox::class_id)
        .def_readwrite("class_name", &object::DetectionBox::class_name)
        .def_readwrite("segmentation", &object::DetectionBox::segmentation)
        .def("__repr__", [](const object::DetectionBox &d)
             { return "<DetectionBox class='" + d.class_name + "' score=" + std::to_string(d.score) + ">"; });

    py::class_<Sam3PromptUnit>(m, "Sam3PromptUnit")
        .def(py::init<>())
        .def(py::init<const std::string &, const std::vector<BoxPrompt> &>(),
             py::arg("text"), py::arg("boxes") = std::vector<BoxPrompt>())
        .def_readwrite("text", &Sam3PromptUnit::text)
        .def_readwrite("boxes", &Sam3PromptUnit::boxes)
        .def("__repr__", [](const Sam3PromptUnit &u) {
            return "<Sam3PromptUnit text='" + u.text + "' boxes_count=" + std::to_string(u.boxes.size()) + ">";
        });

    py::class_<Sam3Input>(m, "Sam3Input")
        .def(py::init<>())
        .def(py::init([](py::array_t<uint8_t> img, const std::vector<Sam3PromptUnit> &prompts, float conf)
                      { return Sam3Input(numpy_to_mat(img), prompts, conf); }),
             py::arg("image"), py::arg("prompts"), py::arg("conf"))
        .def_readwrite("prompts", &Sam3Input::prompts)
        .def_property("image", [](Sam3Input &self)
                      { return mat_to_numpy(self.image); }, [](Sam3Input &self, py::array_t<uint8_t> array)
                      { self.image = numpy_to_mat(array).clone(); });

    // --- 推理类绑定 ---
    py::class_<Sam3Infer, std::shared_ptr<Sam3Infer>>(m, "Sam3Infer")
        .def_static("create_instance",
                    static_cast<std::shared_ptr<Sam3Infer> (*)(const std::string &, const std::string &, const std::string &, const std::string &, int)>(&Sam3Infer::create_instance),
                    py::arg("vision_path"), py::arg("text_path"), py::arg("geometry_path"), py::arg("decoder_path"), py::arg("gpu_id") = 0, 
                    "Create a Sam3Infer instance with all 3 encoders.")
        .def("setup_geometry_input",
             [](Sam3Infer &self, py::array_t<uint8_t> &img, const std::string &label,
                const std::vector<std::pair<std::string, std::array<float, 4>>> &boxes)
                {
                    cv::Mat mat = numpy_to_mat(img).clone();
                    py::gil_scoped_release release;
                    return self.setup_geometry_input(mat, label, boxes);
                },
             py::arg("image"), py::arg("label"), py::arg("boxes"))
        .def("setup_text_inputs", 
            [](Sam3Infer &self, const std::string &text, const std::vector<int64_t> &input_ids, const std::vector<int64_t> &attention_mask)
             {
                if (input_ids.size() != 32 || attention_mask.size() != 32) throw std::runtime_error("Inputs must be 32 elements");
                std::array<int64_t, 32> arr_ids, arr_mask;
                std::copy(input_ids.begin(), input_ids.end(), arr_ids.begin());
                std::copy(attention_mask.begin(), attention_mask.end(), arr_mask.begin());
                self.setup_text_inputs(text, arr_ids, arr_mask); 
            }, 
            py::arg("text"), py::arg("input_ids"), py::arg("attention_mask"))
        .def("forwards",
            [](Sam3Infer &self, const std::vector<Sam3Input> &inputs, const std::string &geom_label, bool return_mask)
            {
                py::gil_scoped_release release;
                return self.forwards(inputs, geom_label, return_mask, nullptr);
            },
            py::arg("inputs"), py::arg("geom_label"), py::arg("return_mask") = false)
        .def("forwards", 
            [](Sam3Infer &self, const std::vector<Sam3Input> &inputs, bool return_mask)
             {
                py::gil_scoped_release release;
                return self.forwards(inputs, return_mask, nullptr); 
            }, 
            py::arg("inputs"), py::arg("return_mask") = false);

    // --- 【新増】OSD 函数绑定 ---

    // 1. 基础 OSD (绘制検出框)
    // 注意：osd 是原地修改图像，为了方便 Python 使用，我们返回修改后的图像
    m.def("osd", [](py::array_t<uint8_t> &img, const std::vector<object::DetectionBox> &boxes, bool osd_rect, double font_scale_ratio) {
        cv::Mat mat = numpy_to_mat(img);
        // 调用 C++ 的 osd 函数，直接修改 mat (同时也修改了 numpy 的内存)
        osd(mat, boxes, osd_rect, font_scale_ratio);
        return img; // 返回原 numpy 数组以便链式调用
    }, py::arg("image"), py::arg("boxes"), py::arg("osd_rect") = true, py::arg("font_scale_ratio") = 0.04,
       "Draw detection boxes, labels and masks on the image in-place.");

    // --- 【新増】Sam3Tracker 追跡クラス绑定 ---

    // TrackedObject 結構
    py::class_<sam3::TrackedObject>(m, "TrackedObject")
        .def(py::init<>())
        .def_readwrite("object_id", &sam3::TrackedObject::object_id)
        .def_readwrite("class_name", &sam3::TrackedObject::class_name)
        .def_readwrite("last_score", &sam3::TrackedObject::last_score)
        .def_readwrite("pointer", &sam3::TrackedObject::pointer)
        .def_readwrite("last_bbox", &sam3::TrackedObject::last_bbox)
        .def_readwrite("is_active", &sam3::TrackedObject::is_active)
        .def_readwrite("last_seen_frame", &sam3::TrackedObject::last_seen_frame)
        .def("__repr__", [](const sam3::TrackedObject &obj) {
            return "<TrackedObject id=" + std::to_string(obj.object_id) +
                   " class='" + obj.class_name + "' active=" + (obj.is_active ? "true" : "false") + ">";
        });

    // TrackingResult 結構
    py::class_<sam3::TrackingResult>(m, "TrackingResult")
        .def(py::init<>())
        .def_readwrite("frame_id", &sam3::TrackingResult::frame_id)
        .def_readwrite("object_id", &sam3::TrackingResult::object_id)
        .def_readwrite("class_name", &sam3::TrackingResult::class_name)
        .def_readwrite("confidence", &sam3::TrackingResult::confidence)
        .def_readwrite("bbox", &sam3::TrackingResult::bbox)
        .def_property("mask",
            [](sam3::TrackingResult &self) { return mat_to_numpy(self.mask); },
            [](sam3::TrackingResult &self, py::array_t<uint8_t> array) {
                self.mask = numpy_to_mat(array).clone();
            })
        .def_readwrite("is_new", &sam3::TrackingResult::is_new)
        .def("__repr__", [](const sam3::TrackingResult &r) {
            return "<TrackingResult frame=" + std::to_string(r.frame_id) +
                   " obj_id=" + std::to_string(r.object_id) +
                   " class='" + r.class_name + "' conf=" + std::to_string(r.confidence) + ">";
        });

    // MemoryBankConfig 結構
    py::class_<sam3::MemoryBankConfig>(m, "MemoryBankConfig")
        .def(py::init<>())
        .def_readwrite("max_recent_frames", &sam3::MemoryBankConfig::max_recent_frames)
        .def_readwrite("max_prompted_frames", &sam3::MemoryBankConfig::max_prompted_frames)
        .def_readwrite("max_objects_per_frame", &sam3::MemoryBankConfig::max_objects_per_frame)
        .def_readwrite("memory_channels", &sam3::MemoryBankConfig::memory_channels)
        .def_readwrite("memory_height", &sam3::MemoryBankConfig::memory_height)
        .def_readwrite("memory_width", &sam3::MemoryBankConfig::memory_width)
        .def_readwrite("pointer_dim", &sam3::MemoryBankConfig::pointer_dim);

    // Sam3TrackerConfig 結構
    py::class_<sam3::Sam3TrackerConfig>(m, "Sam3TrackerConfig")
        .def(py::init<>())
        .def_readwrite("vision_encoder_path", &sam3::Sam3TrackerConfig::vision_encoder_path)
        .def_readwrite("text_encoder_path", &sam3::Sam3TrackerConfig::text_encoder_path)
        .def_readwrite("geometry_encoder_path", &sam3::Sam3TrackerConfig::geometry_encoder_path)
        .def_readwrite("decoder_path", &sam3::Sam3TrackerConfig::decoder_path)
        .def_readwrite("memory_encoder_path", &sam3::Sam3TrackerConfig::memory_encoder_path)
        .def_readwrite("memory_attention_path", &sam3::Sam3TrackerConfig::memory_attention_path)
        .def_readwrite("gpu_id", &sam3::Sam3TrackerConfig::gpu_id)
        .def_readwrite("detection_threshold", &sam3::Sam3TrackerConfig::detection_threshold)
        .def_readwrite("tracking_threshold", &sam3::Sam3TrackerConfig::tracking_threshold)
        .def_readwrite("match_iou_threshold", &sam3::Sam3TrackerConfig::match_iou_threshold)
        .def_readwrite("max_lost_frames", &sam3::Sam3TrackerConfig::max_lost_frames)
        .def_readwrite("memory_config", &sam3::Sam3TrackerConfig::memory_config);

    // Sam3Tracker クラス
    py::class_<sam3::Sam3Tracker, std::shared_ptr<sam3::Sam3Tracker>>(m, "Sam3Tracker")
        .def_static("create_instance", &sam3::Sam3Tracker::create_instance,
            py::arg("config"),
            "Create a Sam3Tracker instance with the given configuration.")
        .def("initialize",
            [](sam3::Sam3Tracker &self, py::array_t<uint8_t> &first_frame,
               const std::vector<Sam3PromptUnit> &prompts, float conf_threshold) {
                cv::Mat mat = numpy_to_mat(first_frame).clone();
                py::gil_scoped_release release;
                return self.initialize(mat, prompts, conf_threshold);
            },
            py::arg("first_frame"), py::arg("prompts"), py::arg("confidence_threshold") = 0.5f,
            "Initialize the tracker with the first frame and prompts.")
        .def("track_frame",
            [](sam3::Sam3Tracker &self, py::array_t<uint8_t> &frame) {
                cv::Mat mat = numpy_to_mat(frame).clone();
                py::gil_scoped_release release;
                return self.track_frame(mat);
            },
            py::arg("frame"),
            "Track objects in the given frame. Returns a list of TrackingResult.")
        .def("add_prompt", &sam3::Sam3Tracker::add_prompt,
            py::arg("prompt"),
            "Add a new prompt to track.")
        .def("remove_object", &sam3::Sam3Tracker::remove_object,
            py::arg("object_id"),
            "Remove an object from tracking.")
        .def("get_tracked_object_ids", &sam3::Sam3Tracker::get_tracked_object_ids,
            "Get list of currently tracked object IDs.")
        .def("get_tracked_objects", &sam3::Sam3Tracker::get_tracked_objects,
            "Get list of currently tracked objects.")
        .def("reset", &sam3::Sam3Tracker::reset,
            "Reset the tracker state.");
}