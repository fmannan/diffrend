#include <iostream>
#include <fstream>
#include <map>

#include <glad/glad.h>

#include "utils.h"
#include "object.h"
#include "shader.h"
#include "scene.h"

#include <glm/gtc/type_ptr.hpp>

void Scene::loadScene(const std::string& filename)
{
    std::ifstream ifs(filename);
    std::string basedir = get_basedir(filename);

    std::cout << "filename: " << filename << std::endl;
    Json::CharReaderBuilder reader;
    Json::Value obj;
    std::string json_err;
    Json::parseFromStream(reader, ifs, &obj, &json_err);

    mVertexShaderPath = basedir + "/" + obj["glsl"]["vertex"].asString();
    mFragmentShaderPath = basedir + "/" + obj["glsl"]["fragment"].asString();

    std::cout << "Vertex shader path: " << mVertexShaderPath << std::endl;
    std::cout << "Fragment shader path: " << mFragmentShaderPath << std::endl;

    mCamera = new Camera(obj["camera"]);

    mColors = loadColors(obj["colors"]);
    loadLights(obj["lights"], mColors);
    loadMaterials(obj["materials"]);

    auto objects_specs = obj["objects"];
    std::cout << "objects: " << objects_specs << std::endl;

    //mObjects.push_back(new TestTriangle());
    for(auto obj: objects_specs["obj"]) {
        std::cout << obj["path"].asString() << std::endl;
        glm::vec3 translate(0.0);
        glm::mat4 rotate(1.0);
        glm::vec3 scale(1.0);
        if(obj["translate"]) {
            std::cout << obj["translate"] << std::endl;
            translate = glm::vec3(obj["translate"][0].asFloat(), obj["translate"][1].asFloat(), obj["translate"][2].asFloat());
        }
        if(obj["scale"]) {
            scale = glm::vec3(obj["scale"][0].asFloat(), obj["scale"][1].asFloat(), obj["scale"][2].asFloat());
        }
        int mat_idx = obj["material_idx"].asInt();
        mObjects.push_back(new TriangleMesh(basedir + "/" + obj["path"].asString(), 
            mMaterials[mat_idx], translate, rotate, scale));
    }
}

std::vector<glm::vec3> Scene::loadColors(const Json::Value& color_table)
{
    std::cout << "Loading Colors" << std::endl;
    std::vector<glm::vec3> colors;
    for(int i = 0; i < color_table.size(); i++) {
        auto clr = color_table[i];
        colors.push_back(glm::vec3(clr[0].asFloat(), clr[1].asFloat(), clr[2].asFloat()));
    }
    return colors;
}

void Scene::loadLights(const Json::Value& light_spec,
    const std::vector<glm::vec3>& colors)
{
    std::cout << "Lights" << std::endl;
    std::cout << light_spec << std::endl;

    auto ambient = light_spec["ambient"];
    mAmbient = glm::vec3(ambient[0].asFloat(), ambient[1].asFloat(), ambient[2].asFloat());
    
    mNumLights = light_spec["pos"].size();
    std::cout << "Number of lights: " << mNumLights << std::endl;
    for(int i = 0; i < mNumLights; i++) {
        auto light_pos = light_spec["pos"][i];
        auto attenuation = light_spec["attenuation"][i];
        mLightPos[i] = glm::vec4(light_pos[0].asFloat(), light_pos[1].asFloat(), light_pos[2].asFloat(), light_pos[3].asFloat());
        mLightColor[i] = colors[light_spec["color_idx"][i].asInt()];
        mLightAttenuation[i] = glm::vec3(attenuation[0].asFloat(), attenuation[1].asFloat(), attenuation[2].asFloat());
        std::cout << mLightColor[i][0] << " " << mLightColor[i][1] << " " << mLightColor[i][2] << std::endl;
        std::cout << mLightAttenuation[i][0] << " " << mLightAttenuation[i][1] << " " << mLightAttenuation[i][2] << std::endl;
    }
}

void Scene::loadMaterials(const Json::Value& material_spec)
{
    auto albedo = material_spec["albedo"];
    auto coeffs = material_spec["coeffs"];
    assert(coeffs.size() == albedo.size());
    for(int i = 0; i < albedo.size(); i++) {
        glm::vec3 mat_albedo = glm::vec3(albedo[i][0].asFloat(), albedo[i][1].asFloat(), albedo[i][2].asFloat());
        glm::vec3 mat_coeffs = glm::vec3(coeffs[i][0].asFloat(), coeffs[i][1].asFloat(), coeffs[i][2].asFloat());
        mMaterials.push_back({mat_albedo, mat_coeffs});
    }
}

void Scene::setup()
{
    mProgram = LoadShaders(mVertexShaderPath, mFragmentShaderPath);

    model_matrix_location = glGetUniformLocation(mProgram, "model");
    view_matrix_location = glGetUniformLocation(mProgram, "view");
    projection_matrix_location = glGetUniformLocation(mProgram, "projection");
    inv_model_view_transpose_location = glGetUniformLocation(mProgram, "inv_model_view_transpose");

    position_location = glGetAttribLocation(mProgram, "position");
    normal_location = glGetAttribLocation(mProgram, "normal");
    albedo_location = glGetAttribLocation(mProgram, "albedo");
    coeffs_location = glGetAttribLocation(mProgram, "coeffs");

    ambient_location = glGetUniformLocation(mProgram, "ambient");
    light_pos_location = glGetUniformLocation(mProgram, "light_pos");
    light_color_location = glGetUniformLocation(mProgram, "light_color");
    light_attenuation_location = glGetUniformLocation(mProgram, "light_attenuation");

    std::cout << "ambient_location : " << ambient_location << std::endl;
    std::cout << "light_attenuation_location : " << light_attenuation_location << std::endl;
    std::map<std::string, GLuint> var_name_map;
    var_name_map["position"] = position_location;
    var_name_map["normal"] = normal_location;
    var_name_map["albedo"] = albedo_location;
    var_name_map["coeffs"] = coeffs_location;
    for(auto obj: mObjects) {
        obj->setup(var_name_map);
    }
}

void Scene::render(const Camera* camera) {
    if(camera == nullptr) {
        camera = mCamera;
    }
    glm::mat4 mModel = glm::mat4(1.0);
    glm::mat4 mView = camera->getViewMatrix();
    glm::mat4 mProjection = camera->getProjectionMatrix();
    glUseProgram(mProgram);
    
    glUniform3fv(ambient_location, 1, glm::value_ptr(mAmbient));
    glUniform4fv(light_pos_location, MAX_NUM_LIGHTS, glm::value_ptr(mLightPos[0]));
    glUniform3fv(light_color_location, MAX_NUM_LIGHTS, glm::value_ptr(mLightColor[0]));
    glUniform3fv(light_attenuation_location, MAX_NUM_LIGHTS, glm::value_ptr(mLightAttenuation[0]));
    glUniformMatrix4fv(view_matrix_location, 1, GL_FALSE, glm::value_ptr(mView));
    glUniformMatrix4fv(projection_matrix_location, 1, GL_FALSE, glm::value_ptr(mProjection));
    for(auto obj: mObjects) {
        glm::mat4 curr_model_tform = mModel * obj->get_transformation();
        glm::mat4 inv_model_view_transpose_tform = glm::transpose(glm::inverse(mView * mModel));
        glUniformMatrix4fv(model_matrix_location, 1, GL_FALSE, glm::value_ptr(curr_model_tform));
        glUniformMatrix4fv(inv_model_view_transpose_location, 1, GL_FALSE, glm::value_ptr(inv_model_view_transpose_tform));
        obj->render(model_matrix_location);
    }
}
