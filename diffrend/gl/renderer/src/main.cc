#include <iostream>
#include <string>
#include <cxxopts/cxxopts.hpp>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#if USE_NATIVE_OSMESA
    #define GLFW_EXPOSE_NATIVE_OSMESA
    #include <GLFW/glfw3native.h>
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include "object.h"
#include "light.h"
#include "utils.h"
#include "scene.h"
#include "renderer.h"
#include "camera.h"


int main(int argc, char** argv) {
    std::cout << "Render Server\nBuild date: " << __DATE__ << " " << __TIME__ << "\n" << std::endl;
    cxxopts::Options options("Render", "Render Server");
    options.add_options()
    ("s,scene", "Scene specification json file", cxxopts::value<std::string>())
    ("t,trajectory", "Trajectory specification json file", cxxopts::value<std::string>())
    ("o,output-dir", "Output directory", cxxopts::value<std::string>())
    ("g,gui", "Interactive mode with GUI", cxxopts::value<bool>());

    auto args = options.parse(argc, argv);

    if(args["scene"].count() == 0) {
        std::cout << "Error: Specify scene file path." << std::endl;
        return -1;
    }

    std::string scene_filename = args["scene"].as<std::string>();
    std::string out_dir = "";
    if(args["output-dir"].count() > 0)
        out_dir = args["output-dir"].as<std::string>();
    if(out_dir.size() > 0)
        out_dir = out_dir + "/";
    bool bGUIMode = args["gui"].as<bool>();

    CameraTrajectory *cam_traj = nullptr;
    std::string trajectory_file = "";
    if(args["trajectory"].count() > 0) {
        trajectory_file = args["trajectory"].as<std::string>();
        cam_traj = new CameraTrajectory(trajectory_file);
    }

    std::cout << "Using scene file: " << scene_filename << std::endl;
    std::cout << "Output directory: " << out_dir << std::endl;

    Scene scene(scene_filename);
    std::cout << "scene width: " << scene.getWidth() << " " << scene.getHeight() << std::endl;
    GLRenderer renderer(&scene, out_dir);
    
    const Camera *camera = nullptr;
    if(bGUIMode) {
        // if in interactive mode
        while(!renderer.shouldClose()) {
            glfwPollEvents();
            if(cam_traj != nullptr) {
                // get next trajectory
                camera = cam_traj->getNext(true);
            }
            renderer.render(camera);
            renderer.swapBuffers();
        }
    } else {
        if(cam_traj != nullptr) {
            while(true) {
                auto cam_fname = cam_traj->getNextCameraAndFilename();
                if(cam_fname.first == nullptr)
                    break;
                renderer.render(cam_fname.first, cam_fname.second);
            }
        } else {
            renderer.render();
        }
    }

    return 0;
}
