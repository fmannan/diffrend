#include <iostream>
#include <fstream>
#include <map>

#include <glad/glad.h>

#include "utils.h"
#include "object.h"
#include "shader.h"

#include <glm/gtc/type_ptr.hpp>
#include <tiny_obj_loader.h>


struct TinyObjIndexComparator {
    using is_transparent = std::true_type;

    bool operator()(const tinyobj::index_t& lhs, const tinyobj::index_t& rhs) const
    {
        //std::cout << "{" << lhs.vertex_index << " " << lhs.normal_index << " " << lhs.texcoord_index << "}, "
        //    << "{" << rhs.vertex_index << " " << rhs.normal_index << " " << rhs.texcoord_index << "}\n";
        if(lhs.vertex_index < rhs.vertex_index)
            return true;

        if(lhs.normal_index < rhs.normal_index)
            return true;

        return lhs.texcoord_index < rhs.texcoord_index;
    }

    bool operator()(const tinyobj::index_t* const lhs, const tinyobj::index_t* const rhs) const
    {
        //std::cout << "{" << lhs->vertex_index << " " << lhs->normal_index << " " << lhs->texcoord_index << "}, "
        //    << "{" << rhs->vertex_index << " " << rhs->normal_index << " " << rhs->texcoord_index << "}\n";
        if(lhs->vertex_index < rhs->vertex_index)
            return true;

        if(lhs->normal_index < rhs->normal_index)
            return true;

        return lhs->texcoord_index < rhs->texcoord_index;
    }
};

glm::vec3 computeNormal(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2)
{
    return glm::normalize(glm::cross(v1 - v0, v2 - v0));
}

void TriangleMesh::loadObj(const std::string& filename,
    Material mat)
{
    mMaterial = mat;
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

    std::vector<glm::vec3> raw_normals;
    for(size_t i = 0; i < attrib.normals.size() / 3; i++) {
        raw_normals.push_back({attrib.normals[i * 3], attrib.normals[i * 3 + 1], attrib.normals[i * 3 + 2]});
    }

    std::vector<glm::vec2> raw_texcoords;
    for(size_t i = 0; i < attrib.texcoords.size() / 2; i++) {
        raw_texcoords.push_back({attrib.texcoords[i * 2], attrib.texcoords[i * 2 + 1]});
    }

    if(raw_normals.size() == 0) { // each face will have a unique normal
        for(size_t i = 0; i < shapes.size(); i++) {
            for(size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
                auto idx = &shapes[i].mesh.indices[3 * f];
                raw_normals.emplace_back(computeNormal(raw_vertices[idx[0].vertex_index],
                    raw_vertices[idx[1].vertex_index], raw_vertices[idx[2].vertex_index]));
                idx[0].normal_index = raw_normals.size() - 1;
                idx[1].normal_index = raw_normals.size() - 1;
                idx[2].normal_index = raw_normals.size() - 1;
            }
        }
    }
    assert(shapes.size() <= 1); // for now, 1 shape per obj file
    //mMaterial.albedo = glm::vec3(0.8); // TODO: FIX
    //mMaterial.coeffs = glm::vec3(1, 0, 0);

    for(size_t i = 0; i < shapes.size(); i++) {
        std::cout << "shapes[i].mesh.indices.size() " << shapes[i].mesh.indices.size() << std::endl;
        for(size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
            // per-face vertex information
            auto idx = &shapes[i].mesh.indices[3 * f];
            for(int k = 0; k < 3; k++) {
                Vertex v;
                v.position = raw_vertices[idx[k].vertex_index];
                v.normal = raw_normals[idx[k].normal_index];
                v.material = mMaterial;
                mBuffer.push_back(v);
            }
        }
    }
    /* // Build VBO and IBO data structures
    std::map<tinyobj::index_t*, int, TinyObjIndexComparator> Idx_LUT;
    for(size_t i = 0; i < shapes.size(); i++) {
        std::cout << "shapes[i].mesh.indices.size() " << shapes[i].mesh.indices.size() << std::endl;
        for(size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
            glm::ivec3 index_buffer_idx;
            for(int k = 0; k < 3; k++) {
                tinyobj::index_t *idx = &shapes[i].mesh.indices[3 * f + k];
                std::cout << "Searching ... " << idx->vertex_index << " " << idx->normal_index << " " << idx->texcoord_index  << std::endl;
                if(Idx_LUT.find(idx) == Idx_LUT.end()) {
                    std::cout << "Assigning " << idx->vertex_index << " " << idx->normal_index << " " << idx->texcoord_index << " ";
                    mVertices.push_back(raw_vertices[idx->vertex_index]);
                    mNormals.push_back(raw_normals[idx->normal_index]);
                    if(idx->texcoord_index >= 0)
                        mTexCoords.push_back(raw_texcoords[idx->texcoord_index]);
                    std::cout << "to index " << mVertices.size() - 1 << " >> ";
                    //Idx_LUT.insert(std::make_pair<tinyobj::index_t, int>(idx, (int)(mVertices.size()) - 1));
                    std::cout << "idx: " << idx << std::endl;
                    //Idx_LUT[idx] = (int)(mVertices.size()) - 1;
                    Idx_LUT.emplace(idx, (int) mVertices.size() - 1);
                    auto find_idx = Idx_LUT.find(idx);
                    std::cout << ((find_idx == Idx_LUT.end()) ? "Not Found" : "Found") << std::endl;
                    assert(find_idx != Idx_LUT.end());
                    std::cout << "check (" << find_idx->first->vertex_index << "," << find_idx->first->normal_index << "," << find_idx->first->texcoord_index << ") " << 
                     find_idx->second << std::endl;
                }
                std::cout << "Using index : " << Idx_LUT[idx] << " for " << idx->vertex_index << " " << idx->normal_index << " " << idx->texcoord_index << std::endl;
                index_buffer_idx[k] = Idx_LUT[idx];
            }
            mIndices.push_back(index_buffer_idx);
        }
    } */
    
    //assert(mNormals.size() == mVertices.size());
    /*for(auto& idx: mIndices) {
        std::cout << idx[0] << " " << idx[1] << " " << idx[2] << std::endl;
    }*/
}

void TriangleMesh::set_transformations(glm::vec3 translate, 
    glm::mat4 rotate, glm::vec3 scale)
{
    mTranslation = translate;
    mRotation = rotate;
    mScale = scale;
}

void TriangleMesh::setup(GLSLVarMap& var_map) {
    GLuint vpos_location = var_map["position"];
    GLuint vnormal_location = var_map["normal"];
    GLuint albedo_location = var_map["albedo"];
    GLuint coeffs_location = var_map["coeffs"];

    GLsizei pos_stride = sizeof(Vertex); //sizeof(glm::vec3) + sizeof(Material);
    GLsizei normal_stride = sizeof(Vertex); //sizeof(Material) + sizeof(glm::vec3);
    GLsizei albedo_stride = sizeof(Vertex); //sizeof(glm::vec3) + sizeof(glm::vec3) + sizeof(glm::vec3);
    GLsizei coeffs_stride = sizeof(Vertex); //albedo_stride;

    glGenVertexArrays(1, &mVAO);
    glBindVertexArray(mVAO);

    glGenBuffers(1, &mVBO);
    glBindBuffer(GL_ARRAY_BUFFER, mVBO);

    std::cout << "Buffer elements: " << mBuffer.size() << std::endl;
    std::cout << "Buffer size: " << sizeof(mBuffer[0]) * mBuffer.size() << std::endl;
    std::cout << "position offset: " << offsetof(Vertex, position) << std::endl;
    std::cout << "normal offset: " << offsetof(Vertex, normal) << std::endl;
    std::cout << "albedo offset: " << offsetof(Vertex, material.albedo) << std::endl;
    std::cout << "coeffs offset: " << offsetof(Vertex, material.coeffs) << std::endl;
    //glBufferData(GL_ARRAY_BUFFER, mVertices.size() * sizeof(glm::vec3), mVertices.data(), GL_STATIC_DRAW);
    glBufferData(GL_ARRAY_BUFFER, mBuffer.size() * sizeof(mBuffer[0]), mBuffer.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(vpos_location);
    glVertexAttribPointer(vpos_location, 3, GL_FLOAT, GL_FALSE,
                          pos_stride, (void*) offsetof(Vertex, position));
    glEnableVertexAttribArray(vnormal_location);
    glVertexAttribPointer(vnormal_location, 3, GL_FLOAT, GL_FALSE,
                          normal_stride, (void*) offsetof(Vertex, normal));
    glEnableVertexAttribArray(albedo_location);
    glVertexAttribPointer(albedo_location, 3, GL_FLOAT, GL_FALSE,
                          albedo_stride, (void*) offsetof(Vertex, material.albedo));
    glEnableVertexAttribArray(coeffs_location);
    glVertexAttribPointer(coeffs_location, 3, GL_FLOAT, GL_FALSE,
                          coeffs_stride, (void*) offsetof(Vertex, material.coeffs));
    // glEnableVertexAttribArray(vcol_location);
    // glVertexAttribPointer(vcol_location, 3, GL_FLOAT, GL_FALSE,
    //                     sizeof(vertices[0]), (void*) (sizeof(float) * 2));
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    /* // Element buffer
    glGenBuffers(1, &mIBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mIndices.size() * sizeof(glm::ivec3), mIndices.data(), GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); */
    glBindVertexArray(0);
}

void TriangleMesh::render(GLint model_matrix_location)
{
    glBindVertexArray(mVAO);
    //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIBO);
    glDrawArrays(GL_TRIANGLES, 0, mBuffer.size() * 3);
    //glDrawElements(GL_TRIANGLES, mIndices.size() * 3, GL_UNSIGNED_INT, (void*) 0);
    glBindVertexArray(0);
}
