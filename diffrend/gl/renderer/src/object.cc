#include <iostream>
#include <fstream>
#include <map>

#include <glad/glad.h>

#include "utils.h"
#include "object.h"
#include "shader.h"

#include <glm/gtc/type_ptr.hpp>
#include <tiny_obj_loader.h>

void TriangleMesh::loadObj(const std::string& filename)
{
    std::string basedir = get_basedir(filename);
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warnings;
    std::string errors;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials,
    &warnings, &errors, filename.c_str(), basedir.c_str());
    std::cout << "LoadObj ret " << ret << std::endl;
    std::cout << "warnings " << warnings << std::endl;
    std::cout << "errors " << errors << std::endl;
    std::cout << "num vertices: " << attrib.vertices.size() << std::endl;
    std::cout << "num normals: " << attrib.normals.size() << std::endl;
    std::cout << "num shapes: " << shapes.size() << std::endl;

    std::vector<glm::vec3> raw_vertices;
    for(size_t i = 0; i < attrib.vertices.size() / 3; i++) {
        raw_vertices.push_back({attrib.vertices[i * 3], attrib.vertices[i * 3 + 1], attrib.vertices[i * 3 + 2]});
    }

    for(size_t i = 0; i < shapes.size(); i++) {
        std::cout << "shapes[i].mesh.indices.size() " << shapes[i].mesh.indices.size() << std::endl;
        for(size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
            auto idx = &shapes[i].mesh.indices[3 * f];
            //std::cout << idx[0].vertex_index << " " << idx[1].vertex_index << " " << idx[2].vertex_index << std::endl;
            mIndices.push_back({idx[0].vertex_index, idx[1].vertex_index, idx[2].vertex_index});
            mVertices.push_back(raw_vertices[idx[0].vertex_index]);
            mVertices.push_back(raw_vertices[idx[1].vertex_index]);
            mVertices.push_back(raw_vertices[idx[2].vertex_index]);
            for(int k = 0; k < 3; k++) {
                for(int j = 0; j < 3; j++) {
                    //std::cout << raw_vertices[idx[k].vertex_index][j] << " ";
                    mBuffer.push_back(raw_vertices[idx[k].vertex_index][j]);
                }
            }
         }
    }
    // std::cout << __FUNCTION__ << " mVertices.size() " << mVertices.size() << std::endl;
    // for(auto v: mVertices) {
    //     std::cout << v[0] << " " << v[1] << " " << v[2] << std::endl;
    // }
}

void TriangleMesh::setup(GLSLVarMap& var_map) {
    GLuint vpos_location = var_map["position"];
    GLuint albedo_location = var_map["albedo"];
    size_t buffer_size = mBuffer.size() * sizeof(float); //mVertices.size() * 3 * sizeof(float);
    std::cout << "[TriangleMesh::setup] buffer_size: " << buffer_size << std::endl;
    glGenVertexArrays(1, &mVao);
    glBindVertexArray(mVao);

    glGenBuffers(1, &mVbo);
    glBindBuffer(GL_ARRAY_BUFFER, mVbo);

    glBufferData(GL_ARRAY_BUFFER, buffer_size, mBuffer.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(vpos_location);
    glVertexAttribPointer(vpos_location, 3, GL_FLOAT, GL_FALSE,
                          0, (void*) 0);
    // glEnableVertexAttribArray(vcol_location);
    // glVertexAttribPointer(vcol_location, 3, GL_FLOAT, GL_FALSE,
    //                     sizeof(vertices[0]), (void*) (sizeof(float) * 2));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void TriangleMesh::render()
{
    glBindVertexArray(mVao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
}
