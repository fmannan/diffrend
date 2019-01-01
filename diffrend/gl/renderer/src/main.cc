#include <iostream>
//#include <sstream>
#include <string>
//#include <map>
#include <cxxopts/cxxopts.hpp>
//#include <json/json.h>
//#include <npy/npy.hpp>

// #include <glm/glm.hpp>
// #include <glm/gtc/matrix_transform.hpp>
// #include <glm/gtc/type_ptr.hpp>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#if USE_NATIVE_OSMESA
    #define GLFW_EXPOSE_NATIVE_OSMESA
    #include <GLFW/glfw3native.h>
#endif

//#include <linmath.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

//#include <tiny_obj_loader.h>
#include "object.h"
#include "light.h"
#include "utils.h"
#include "scene.h"
#include "renderer.h"

int main(int argc, char** argv) {
    std::cout << "Render Server\nBuild date: " << __DATE__ << " " << __TIME__ << "\n" << std::endl;
    cxxopts::Options options("Render", "Render Server");
    options.add_options()
    ("s,scene", "Scene specification json file", cxxopts::value<std::string>())
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
    bool bGUIMode = args["gui"].as<bool>();

    std::cout << "Using scene file: " << scene_filename << std::endl;
    std::cout << "Output directory: " << out_dir << std::endl;

    Scene scene(scene_filename);
    std::cout << "scene width: " << scene.getWidth() << " " << scene.getHeight() << std::endl;
    GLRenderer renderer(&scene, out_dir); // remove specifying width, height? where should this be specified?
    
    if(bGUIMode) {
        // if in interactive mode
        while(!renderer.shouldClose()) {
            glfwPollEvents();
            renderer.render();
            renderer.swapBuffers();
        }
    } else {
        renderer.render();
    }

    return 0;
}
