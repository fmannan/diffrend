#pragma once

#include <string>
#include <memory>
#include <json/json.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#if USE_NATIVE_OSMESA
    #define GLFW_EXPOSE_NATIVE_OSMESA
    #include <GLFW/glfw3native.h>
#endif

#include "camera.h"
#include "light.h"

#define MAX_NUM_LIGHTS 8

class Scene {
public:
    Scene(const std::string& filename) {
        loadScene(filename);
    }

    void setup();
    void render();

    int getWidth() { return mCamera->getWidth(); }
    int getHeight() { return mCamera->getHeight(); }
private:
    void loadScene(const std::string& filename);
    std::vector<glm::vec3> loadColors(const Json::Value& color_table);
    void loadLights(const Json::Value& light_spec,
        const std::vector<glm::vec3>& colors);
    void loadMaterials(const Json::Value& material_spec);

    std::vector<GLRenderableObject*> mObjects;
    //std::vector<std::shared_ptr<Light>> mLights;
    glm::vec3 mAmbient;
    std::vector<Material> mMaterials;
    int mNumLights;
    glm::vec4 mLightPos[MAX_NUM_LIGHTS];
    glm::vec3 mLightColor[MAX_NUM_LIGHTS];
    glm::vec3 mLightAttenuation[MAX_NUM_LIGHTS];
    std::vector<glm::vec3> mColors;

    Camera* mCamera;
    GLuint mProgram;
    GLint model_matrix_location, view_matrix_location, projection_matrix_location;
    GLint position_location, normal_location, albedo_location, coeffs_location;
    GLint ambient_location;
    GLint light_pos_location;
    GLint light_attenuation_location;
    GLint light_color_location;
    GLint num_lights;
    
    std::string mVertexShaderPath;
    std::string mFragmentShaderPath;
};

