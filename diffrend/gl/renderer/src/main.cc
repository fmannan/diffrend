#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <cxxopts/cxxopts.hpp>
#include <json/json.h>
#include <npy/npy.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#if USE_NATIVE_OSMESA
    #define GLFW_EXPOSE_NATIVE_OSMESA
    #include <GLFW/glfw3native.h>
#endif

#include <linmath.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <tiny_obj_loader.h>

int main(int argc, char** argv) {
    std::cout << "Render Server\nBuild date: " << __DATE__ << " " << __TIME__ << "\n" << std::endl;
    cxxopts::Options options("Render", "Render Server");
    options.add_options()
    ("s,scene", "Scene specification json file", cxxopts::value<std::string>())
    ("o,output-dir", "Output directory", cxxopts::value<std::string>());

    auto args = options.parse(argc, argv);
    std::string scene_filename = args["scene"].as<std::string>();
    std::string out_dir = args["output-dir"].as<std::string>();

    std::cout << "Using scene file: " << scene_filename << std::endl;
    std::cout << "Output directory: " << out_dir << std::endl;

    return 0;
}
