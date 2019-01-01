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

    auto objects_specs = obj["objects"];
    std::cout << "objects: " << objects_specs << std::endl;

    //mObjects.push_back(new TestTriangle());
    for(auto obj: objects_specs["obj"]) {
        std::cout << obj["path"].asString() << std::endl;
        mObjects.push_back(new TriangleMesh(basedir + "/" + obj["path"].asString()));
    }
}

void Scene::setup()
{
    mProgram = LoadShaders(mVertexShaderPath, mFragmentShaderPath);

    model_matrix_location = glGetUniformLocation(mProgram, "model");
    view_matrix_location = glGetUniformLocation(mProgram, "view");
    projection_matrix_location = glGetUniformLocation(mProgram, "projection");

    position_location = glGetAttribLocation(mProgram, "position");
    albedo_location = glGetAttribLocation(mProgram, "albedo");

    std::map<std::string, GLuint> var_name_map;
    var_name_map["position"] = position_location;
    var_name_map["albedo"] = albedo_location;
    for(auto obj: mObjects) {
        obj->setup(var_name_map);
    }
}

void Scene::render() {
    glm::mat4 mModel = glm::mat4(1.0);
    glm::mat4 mView = mCamera->getViewMatrix();
    glm::mat4 mProjection = mCamera->getProjectionMatrix();
    glUseProgram(mProgram);
    glUniformMatrix4fv(model_matrix_location, 1, GL_FALSE, glm::value_ptr(mModel));
    glUniformMatrix4fv(view_matrix_location, 1, GL_FALSE, glm::value_ptr(mView));
    glUniformMatrix4fv(projection_matrix_location, 1, GL_FALSE, glm::value_ptr(mProjection));
    for(auto obj: mObjects) {
        obj->render();
    }
}
